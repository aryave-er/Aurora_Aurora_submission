import sys, torch, numpy as np
sys.path.insert(0, 'src')
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from PIL import Image
from models import get_swin_transformer, LabelSmoothingCrossEntropy, SWAWrapper
from dataset import CLASS_TO_IDX

Path('outputs').mkdir(exist_ok=True)
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Device: {DEVICE}')

files, labels = [], []
for root in ['Data/EuroSAT/train', 'Data/EuroSAT/val']:
    for class_dir in sorted(Path(root).iterdir()):
        if not class_dir.is_dir(): continue
        label = CLASS_TO_IDX.get(class_dir.name)
        if label is None: continue
        for f in sorted(class_dir.iterdir()):
            if f.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                files.append(f)
                labels.append(label)

print(f'Total RGB files: {len(files)}')

np.random.seed(42)
idx   = np.random.permutation(len(files))
n_val = int(0.2 * len(files))
train_idx, val_idx = idx[n_val:], idx[:n_val]
train_files = [files[i] for i in train_idx]

print('Computing RGB stats from train split...')
running_mean = np.zeros(3, dtype=np.float64)
running_var  = np.zeros(3, dtype=np.float64)
n_pixels = 0
for f in tqdm(train_files, desc='Stats', leave=False):
    img = np.array(Image.open(f).convert('RGB')).astype(np.float64) / 255.0
    pixels = img.reshape(-1, 3).T  # (3, N)
    running_mean += pixels.sum(axis=1)
    running_var  += (pixels ** 2).sum(axis=1)
    n_pixels     += pixels.shape[1]

MEAN = (running_mean / n_pixels).astype(np.float32)
STD  = np.sqrt(running_var / n_pixels - (running_mean / n_pixels) ** 2).astype(np.float32)
STD  = np.maximum(STD, 1e-6)
print(f'Mean: {MEAN.tolist()}')
print(f'Std:  {STD.tolist()}')

np.save('outputs/rgb_means.npy', MEAN)
np.save('outputs/rgb_stds.npy',  STD)

class RGBDataset(Dataset):
    def __init__(self, files, labels, is_train=True):
        self.files, self.labels, self.is_train = files, labels, is_train
        self.train_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(360),
            transforms.ToTensor(),
            transforms.Normalize(MEAN.tolist(), STD.tolist()),
        ])
        self.val_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN.tolist(), STD.tolist()),
        ])
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        tf  = self.train_tf if self.is_train else self.val_tf
        return tf(img), self.labels[idx]

train_ds = RGBDataset(train_files, [labels[i] for i in train_idx], is_train=True)
val_ds   = RGBDataset([files[i] for i in val_idx], [labels[i] for i in val_idx], is_train=False)
print(f'Train: {len(train_ds)} | Val: {len(val_ds)}')

train_dl = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=0, pin_memory=False)
val_dl   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=0, pin_memory=False)

model = get_swin_transformer(num_classes=10, pretrained=True).to(DEVICE)
print(f'Swin-T: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params')

EPOCHS    = 40
SWA_START = 30
criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
swa       = SWAWrapper(model, DEVICE)
best_acc  = 0.0

print(f'Training Swin for {EPOCHS} epochs (SWA from epoch {SWA_START})...\n')

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
    if epoch >= SWA_START:
        swa.update()

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
        torch.save(model.state_dict(), 'outputs/swin_transformer_rgb_best.pth')
        marker = ' ◀ best'
    print(f'Epoch {epoch:02d}/{EPOCHS}  train={tl:.4f}/{ta:.4f}  val={va:.4f}  best={best_acc:.4f}{marker}', flush=True)

if swa.n_updates > 0:
    print(f'\nFinalizing SWA ({swa.n_updates} checkpoints)...')
    swa_model = swa.finalize(train_dl)
    swa_model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels_b in val_dl:
            images, labels_b = images.to(DEVICE), labels_b.to(DEVICE)
            correct += (swa_model(images).argmax(1) == labels_b).sum().item()
            total   += labels_b.size(0)
    swa_acc = correct / total
    print(f'SWA val accuracy: {swa_acc*100:.2f}%')
    torch.save(swa_model.state_dict(), 'outputs/swin_transformer_rgb_swa.pth')
    if swa_acc > best_acc:
        best_acc = swa_acc
        print('SWA is best model.')

print(f'\nDone. Best val accuracy: {best_acc*100:.2f}%')