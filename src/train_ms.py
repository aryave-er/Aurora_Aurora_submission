import sys, torch, numpy as np, rasterio
sys.path.insert(0, 'src')
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from models import construct_multispectral_resnet50, LabelSmoothingCrossEntropy
from dataset import CLASS_TO_IDX

Path('outputs').mkdir(exist_ok=True)
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Device: {DEVICE}')

files, labels = [], []
for root in ['Data/EuroSATallBands/train', 'Data/EuroSATallBands/val']:
    for class_dir in sorted(Path(root).iterdir()):
        if not class_dir.is_dir(): continue
        label = CLASS_TO_IDX.get(class_dir.name)
        if label is None: continue
        for f in sorted(class_dir.glob('*.tif')):
            files.append(f)
            labels.append(label)

print(f'Total files: {len(files)}')

np.random.seed(42)
idx   = np.random.permutation(len(files))
n_val = int(0.2 * len(files))
train_idx, val_idx = idx[n_val:], idx[:n_val]
train_files = [files[i] for i in train_idx]

print('Computing global stats from train split (one-time, ~5 min)...')
running_mean = np.zeros(13, dtype=np.float64)
running_var  = np.zeros(13, dtype=np.float64)
n_pixels = 0

for f in tqdm(train_files, desc='Stats'):
    with rasterio.open(f) as src:
        raw = src.read().astype(np.float64)  # (13, 64, 64)
    pixels = raw.reshape(13, -1)
    running_mean += pixels.sum(axis=1)
    running_var  += (pixels ** 2).sum(axis=1)
    n_pixels     += pixels.shape[1]

MEANS = (running_mean / n_pixels).astype(np.float32)
STDS  = np.sqrt(running_var / n_pixels - (running_mean / n_pixels) ** 2).astype(np.float32)
STDS  = np.maximum(STDS, 1.0)

np.save('outputs/ms_means.npy', MEANS)
np.save('outputs/ms_stds.npy',  STDS)
print('Means:', MEANS.round(1).tolist())
print('Stds: ', STDS.round(1).tolist())

class MSDataset(Dataset):
    def __init__(self, files, labels, is_train=True):
        self.files    = files
        self.labels   = labels
        self.is_train = is_train
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        with rasterio.open(self.files[idx]) as src:
            raw = src.read().astype(np.float32)
        normalized = (raw - MEANS[:, None, None]) / (STDS[:, None, None] + 1e-8)
        if self.is_train:
            if np.random.rand() > 0.5: normalized = np.flip(normalized, axis=2).copy()
            if np.random.rand() > 0.5: normalized = np.flip(normalized, axis=1).copy()
            normalized = np.rot90(normalized, k=np.random.randint(4), axes=(1,2)).copy()
        return torch.from_numpy(normalized), self.labels[idx]

train_ds = MSDataset(train_files, [labels[i] for i in train_idx], is_train=True)
val_ds   = MSDataset([files[i] for i in val_idx], [labels[i] for i in val_idx], is_train=False)
print(f'Train: {len(train_ds)} | Val: {len(val_ds)}')

train_dl = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=0, pin_memory=False)
val_dl   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=0, pin_memory=False)

model     = construct_multispectral_resnet50(num_classes=10, in_channels=13).to(DEVICE)
criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35, eta_min=1e-6)
print(f'Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M\n')

EPOCHS, best_acc = 35, 0.0
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = total_correct = total_samples = 0
    for images, labels_b in tqdm(train_dl, desc=f'Epoch {epoch:02d}/{EPOCHS}', leave=False):
        images, labels_b = images.to(DEVICE), labels_b.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        bs = images.size(0)
        total_loss    += loss.item() * bs
        total_samples += bs
        total_correct += (outputs.argmax(1) == labels_b).sum().item()
    scheduler.step()
    tl = total_loss / total_samples
    ta = total_correct / total_samples

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels_b in val_dl:
            images, labels_b = images.to(DEVICE), labels_b.to(DEVICE)
            correct += (model(images).argmax(1) == labels_b).sum().item()
            total   += labels_b.size(0)
    va = correct / total

    marker = ''
    if va > best_acc:
        best_acc = va
        torch.save(model.state_dict(), 'outputs/resnet50_ms_surgery_best.pth')
        marker = ' ◀ best'
    print(f'Epoch {epoch:02d}/{EPOCHS}  train={tl:.4f}/{ta:.4f}  val={va:.4f}  best={best_acc:.4f}{marker}', flush=True)

print(f'\nDone. Best val accuracy: {best_acc*100:.2f}%')