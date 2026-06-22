"""
Aurora — Geo Snap Paradigm
explainability.py: Task 2 — Explainability & Model Interpretation

Generates:
  outputs/xai/gradcam_correct.png      — Grad-CAM on correct Swin-T predictions
  outputs/xai/gradcam_incorrect.png    — Grad-CAM on incorrect predictions
  outputs/xai/gradcam_nearmiss.png     — Low-confidence correct predictions (only if val errors = 0)
  outputs/xai/confusion_matrix.png     — Confusion matrix (val set, Swin-T RGB)
  outputs/xai/calibration.png          — Reliability diagram (val sample)
  outputs/xai/band_importance.png      — Occlusion-based band importance (MS ResNet-50)
  outputs/xai/band_importance.csv      — Band importance values
  outputs/xai/spectral_signatures.png  — Mean spectral profile per class

Run from project root:
    python3 -W ignore explainability.py
"""

import sys, warnings
sys.path.insert(0, 'src')
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
import csv

# pyrefly: ignore [missing-import]
from models import (    
    get_swin_transformer,
    construct_multispectral_resnet50,
    get_target_layer,
)
# pyrefly: ignore [missing-import]
from dataset import CLASS_TO_IDX, IDX_TO_CLASS, BAND_ORDER

DEVICE      = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
OUT_DIR     = Path('outputs/xai')
OUT_DIR.mkdir(parents=True, exist_ok=True)

RGB_CKPT    = 'outputs/swin_transformer_rgb_best.pth'
MS_CKPT     = 'outputs/resnet50_ms_surgery_best.pth'

RGB_VAL_DIR = Path('Data/EuroSAT/val')
MS_VAL_DIR  = Path('Data/EuroSATallBands/val')

# Class names in model order (alphabetical — matches CLASS_TO_IDX)
CLASS_NAMES = [IDX_TO_CLASS[i] for i in range(10)]
N_CLASSES   = 10

BAND_DESC = {
    'B1':'Aerosol', 'B2':'Blue', 'B3':'Green', 'B4':'Red',
    'B5':'VRE1', 'B6':'VRE2', 'B7':'VRE3', 'B8':'NIR',
    'B8A':'NNIR', 'B9':'H₂O', 'B10':'Cirrus', 'B11':'SWIR1', 'B12':'SWIR2'
}

SHORT_NAMES = {
    'AnnualCrop': 'AnnualCrop', 'Forest': 'Forest',
    'HerbaceousVegetation': 'HerbVeg', 'Highway': 'Highway',
    'Industrial': 'Industrial', 'Pasture': 'Pasture',
    'PermanentCrop': 'PermCrop', 'Residential': 'Resident',
    'River': 'River', 'SeaLake': 'SeaLake',
}
SHORT_LIST  = [SHORT_NAMES[c] for c in CLASS_NAMES]

print(f'Device: {DEVICE}')
print(f'Class order: {CLASS_NAMES}')

try:
    RGB_MEAN = np.load('outputs/rgb_means.npy').tolist()
    RGB_STD  = np.load('outputs/rgb_stds.npy').tolist()
    print(f'Loaded RGB stats from outputs/')
except FileNotFoundError:
    RGB_MEAN = [0.3444, 0.3803, 0.4078]   # EuroSAT approximate
    RGB_STD  = [0.2025, 0.1364, 0.1148]
    print('Using fallback RGB stats')

try:
    MS_MEANS = np.load('outputs/ms_means.npy')
    MS_STDS  = np.load('outputs/ms_stds.npy')
    print('Loaded MS stats from outputs/')
except FileNotFoundError:
    MS_MEANS = np.array([1345., 1103., 1019.,  915., 1166., 1960., 2321., 2255.,
                          726.,   12., 1794., 1096., 2549.], dtype=np.float32)
    MS_STDS  = np.array([  60.,  144.,  175.,  256.,  209.,  337.,  429.,  507.,
                            95.,    1.,  361.,  284.,  475.], dtype=np.float32)
    print('Using fallback MS stats')


# ══════════════════════════════════════════════════════════════════════════════
# 1. Load models
# ══════════════════════════════════════════════════════════════════════════════

print('\n── Loading Swin-T (RGB) ──')
rgb_model = get_swin_transformer(num_classes=10, pretrained=False).to(DEVICE)
rgb_model.load_state_dict(torch.load(RGB_CKPT, map_location=DEVICE))
rgb_model.eval()
print('  OK')

print('── Loading ResNet-50 MS ──')
ms_model = construct_multispectral_resnet50(num_classes=10, in_channels=13).to(DEVICE)
ms_model.load_state_dict(torch.load(MS_CKPT, map_location=DEVICE))
ms_model.eval()
print('  OK')


# ══════════════════════════════════════════════════════════════════════════════
# 2. Collect val images
# ══════════════════════════════════════════════════════════════════════════════

val_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(RGB_MEAN, RGB_STD),
])

def collect_rgb_val(max_per_class=30):
    """Returns list of (path, label_idx)."""
    items = []
    for cls_name, idx in CLASS_TO_IDX.items():
        cls_dir = RGB_VAL_DIR / cls_name
        if not cls_dir.exists():
            print(f'  WARNING: {cls_dir} not found')
            continue
        files = sorted(cls_dir.iterdir())[:max_per_class]
        for f in files:
            if f.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                items.append((f, idx))
    return items

print('\n── Collecting RGB val images ──')
rgb_val = collect_rgb_val(max_per_class=30)
print(f'  {len(rgb_val)} images across {len(set(x[1] for x in rgb_val))} classes')


# ══════════════════════════════════════════════════════════════════════════════
# 3. Run inference → build correct / incorrect lists
# ══════════════════════════════════════════════════════════════════════════════

print('\n── Running RGB inference on val set ──')
correct_samples   = []   # (path, true_idx, pred_idx, prob)
incorrect_samples = []

all_true, all_pred = [], []

for path, true_idx in tqdm(rgb_val, leave=False):
    img   = Image.open(path).convert('RGB')
    tensor = val_tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = rgb_model(tensor)
        probs  = F.softmax(logits, dim=1)[0]
    pred_idx = probs.argmax().item()
    conf     = probs[pred_idx].item()

    all_true.append(true_idx)
    all_pred.append(pred_idx)

    entry = (path, true_idx, pred_idx, conf)
    if pred_idx == true_idx:
        correct_samples.append(entry)
    else:
        incorrect_samples.append(entry)

val_acc = len(correct_samples) / len(rgb_val)
print(f'  Val accuracy on sample: {val_acc*100:.1f}%')
print(f'  Correct: {len(correct_samples)}  |  Incorrect: {len(incorrect_samples)}')



# ══════════════════════════════════════════════════════════════════════════════
# 4. Confusion matrix
# ══════════════════════════════════════════════════════════════════════════════

print('\n── Plotting confusion matrix ──')

cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
for t, p in zip(all_true, all_pred):
    cm[t, p] += 1

cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax.set_xticks(range(N_CLASSES))
ax.set_yticks(range(N_CLASSES))
ax.set_xticklabels(SHORT_LIST, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(SHORT_LIST, fontsize=9)
ax.set_xlabel('Predicted Label', fontsize=11)
ax.set_ylabel('True Label', fontsize=11)
ax.set_title('Swin-T RGB — Confusion Matrix (Val Set)', fontsize=12, fontweight='bold')

for i in range(N_CLASSES):
    for j in range(N_CLASSES):
        val = cm[i, j]
        if val > 0:
            color = 'white' if cm_norm[i, j] > 0.6 else 'black'
            ax.text(j, i, str(val), ha='center', va='center', fontsize=7, color=color)

plt.tight_layout()
fig.savefig(OUT_DIR / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'  Saved → {OUT_DIR}/confusion_matrix.png')

print('\n  Top misclassifications:')
off_diag = [(cm[i,j], CLASS_NAMES[i], CLASS_NAMES[j])
            for i in range(N_CLASSES) for j in range(N_CLASSES) if i != j and cm[i,j] > 0]
for count, true_cls, pred_cls in sorted(off_diag, reverse=True)[:8]:
    print(f'    {true_cls:22s} → {pred_cls:22s}  ({count}x)')


# ══════════════════════════════════════════════════════════════════════════════
# 5. Grad-CAM implementation (no external library needed)
# ══════════════════════════════════════════════════════════════════════════════

class GradCAM:
    """Simple Grad-CAM using forward/backward hooks."""
    def __init__(self, model, target_layer):
        self.model        = model
        self.activations  = None
        self.gradients    = None
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, tensor, class_idx=None):
        self.model.zero_grad()
        logits = self.model(tensor)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        score = logits[0, class_idx]
        score.backward()

        # Pool gradients over spatial dims
        grads = self.gradients                     # (1, C, H, W) or (1, C, L) for Swin
        acts  = self.activations                   # same shape

        if grads.dim() == 4:
            weights = grads.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
            cam = (weights * acts).sum(dim=1, keepdim=True)   # (1, 1, H, W)
            cam = F.relu(cam)
            cam = cam.squeeze().cpu().numpy()
        else:
            # Swin: tokens → average over token dim
            weights = grads.mean(dim=(1,), keepdim=True)
            cam     = (weights * acts).sum(dim=-1)
            cam     = F.relu(cam).squeeze().cpu().numpy()
            # reshape to square
            side = int(cam.shape[-1] ** 0.5)
            cam  = cam.reshape(side, side) if cam.ndim == 1 else cam

        # Normalize
        cam -= cam.min()
        cam_max = cam.max()
        if cam_max > 0:
            cam /= cam_max
        return cam, logits.softmax(dim=1)[0].detach().cpu().numpy()

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def overlay_cam(img_pil, cam, alpha=0.45):
    """Blend Grad-CAM heatmap onto image. Returns RGB numpy array."""
    img_np = np.array(img_pil.resize((64, 64)))
    cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((64, 64),
                                                                                 Image.BILINEAR)) / 255.0
    cmap = plt.cm.jet
    heatmap = (cmap(cam_resized)[:, :, :3] * 255).astype(np.uint8)
    blended = (alpha * heatmap + (1 - alpha) * img_np).astype(np.uint8)
    return blended


# ══════════════════════════════════════════════════════════════════════════════
# 6. Generate Grad-CAM panels
# ══════════════════════════════════════════════════════════════════════════════

print('\n── Generating Grad-CAM heatmaps ──')
target_layer = get_target_layer(rgb_model, 'swin')
gcam = GradCAM(rgb_model, target_layer)

def gradcam_panel(samples, title, filename, n_show=12):
    """
    samples: list of (path, true_idx, pred_idx, conf)
    Shows: original | heatmap | overlay for each sample
    """
    if len(samples) == 0:
        print(f'  Skipping {filename} — no samples')
        return

    picks = samples[:n_show]
    n     = len(picks)
    cols  = min(n, 4)
    rows  = (n + cols - 1) // cols

    fig = plt.figure(figsize=(cols * 4.5, rows * 3.8))
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)

    for i, (path, true_idx, pred_idx, conf) in enumerate(picks):
        img   = Image.open(path).convert('RGB')
        tensor = val_tf(img).unsqueeze(0).to(DEVICE)
        tensor.requires_grad_(False)

        cam, probs = gcam(tensor, class_idx=pred_idx)
        overlay = overlay_cam(img, cam)

        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(overlay)
        true_name = SHORT_NAMES[CLASS_NAMES[true_idx]]
        pred_name = SHORT_NAMES[CLASS_NAMES[pred_idx]]
        color = '#2ecc71' if true_idx == pred_idx else '#e74c3c'
        ax.set_title(f'True: {true_name}\nPred: {pred_name} ({conf:.1%})',
                     fontsize=7.5, color=color, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved → {OUT_DIR}/{filename}')

diverse_correct = []
seen = {}
for s in correct_samples:
    cls = s[1]
    if seen.get(cls, 0) < 2:
        diverse_correct.append(s)
        seen[cls] = seen.get(cls, 0) + 1

gradcam_panel(diverse_correct,
              'Grad-CAM — Correct Predictions (Swin-T RGB)',
              'gradcam_correct.png', n_show=12)

incorrect_sorted = sorted(incorrect_samples, key=lambda x: -x[3])
gradcam_panel(incorrect_sorted,
              'Grad-CAM — Incorrect Predictions (Swin-T RGB)\n'
              '(Red title = wrong, sorted by model confidence)',
              'gradcam_incorrect.png', n_show=min(12, len(incorrect_sorted)))
if len(incorrect_samples) == 0:
    print('  No errors on val sample — using lowest-confidence correct predictions as near-misses')
    near_misses = sorted(correct_samples, key=lambda x: x[3])[:12]
    gradcam_panel(near_misses,
                  'Grad-CAM — Low-Confidence Predictions (Swin-T RGB)\n'
                  '(Correct but uncertain — potential confusion cases)',
                  'gradcam_nearmiss.png', n_show=12)
gcam.remove_hooks()

# ── Confidence calibration ─────────────────────────────────────────────────────
print('\n── Confidence calibration ──')
all_confs  = [s[3] for s in correct_samples] + [s[3] for s in incorrect_samples]
all_flags  = [1]*len(correct_samples) + [0]*len(incorrect_samples)

bins = np.linspace(0.5, 1.0, 11)
bin_acc, bin_conf, bin_count = [], [], []
for i in range(len(bins)-1):
    mask = [bins[i] <= c < bins[i+1] for c in all_confs]
    if not any(mask): continue
    bin_acc.append(np.mean([all_flags[j]  for j,m in enumerate(mask) if m]))
    bin_conf.append(np.mean([all_confs[j] for j,m in enumerate(mask) if m]))
    bin_count.append(sum(mask))

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0.5, 1.0], [0.5, 1.0], 'k--', lw=1.5, label='Perfect calibration')
ax.scatter(bin_conf, bin_acc, s=[c*2 for c in bin_count], zorder=3, label='Model bins')
ax.set_xlim(0.5, 1.0); ax.set_ylim(0.5, 1.05)
ax.set_xlabel('Mean Confidence'); ax.set_ylabel('Fraction Correct')
ax.set_title('Reliability Diagram — Swin-T RGB (Val Sample)', fontweight='bold')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(OUT_DIR / 'calibration.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'  Saved → {OUT_DIR}/calibration.png')

# ══════════════════════════════════════════════════════════════════════════════
# 7. Band importance via occlusion (MS model)
# ══════════════════════════════════════════════════════════════════════════════

print('\n── Computing band importance (MS ResNet-50) ──')

try:
    import rasterio
    RASTERIO_OK = True
except ImportError:
    RASTERIO_OK = False
    print('  WARNING: rasterio not available — skipping band importance')

if RASTERIO_OK:
    def load_ms_tensor(path):
        with rasterio.open(path) as src:
            raw = src.read().astype(np.float32)   # (13, 64, 64)
        normed = (raw - MS_MEANS[:, None, None]) / (MS_STDS[:, None, None] + 1e-8)
        return torch.from_numpy(normed).unsqueeze(0)   # (1, 13, 64, 64)

    ms_items = []
    for cls_name, idx in CLASS_TO_IDX.items():
        cls_dir = MS_VAL_DIR / cls_name
        if not cls_dir.exists():
            continue
        tifs = sorted(cls_dir.glob('*.tif'))[:15]
        for f in tifs:
            ms_items.append((f, idx))

    print(f'  MS val sample: {len(ms_items)} images')

    baseline_correct = 0
    band_drop      = np.zeros(13)
    band_conf_drop = np.zeros(13)
    baseline_confs = []

    print('  Computing baseline...')
    for path, true_idx in tqdm(ms_items, leave=False):
        t = load_ms_tensor(path).to(DEVICE)
        with torch.no_grad():
            probs = F.softmax(ms_model(t), dim=1)[0]
        pred = probs.argmax().item()
        baseline_correct += int(pred == true_idx)
        baseline_confs.append(probs[true_idx].item())

    baseline_acc  = baseline_correct / len(ms_items)
    baseline_conf = np.mean(baseline_confs)
    print(f'  Baseline acc={baseline_acc*100:.1f}%  mean true-class conf={baseline_conf*100:.1f}%')

    print('  Occlusion per band...')
    for b_idx in tqdm(range(13), desc='Bands', leave=False):
        drop_correct = 0
        drop_confs   = []
        for path, true_idx in ms_items:
            t = load_ms_tensor(path).to(DEVICE)
            t[:, b_idx, :, :] = 0.0    # zero-out band b_idx
            with torch.no_grad():
                probs = F.softmax(ms_model(t), dim=1)[0]
            pred = probs.argmax().item()
            drop_correct  += int(pred == true_idx)
            drop_confs.append(probs[true_idx].item())
        occluded_acc  = drop_correct / len(ms_items)
        occluded_conf = np.mean(drop_confs)
        band_drop[b_idx]      = baseline_acc  - occluded_acc
        band_conf_drop[b_idx] = baseline_conf - occluded_conf

    # Save CSV
    csv_path = OUT_DIR / 'band_importance.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['band', 'acc_drop', 'conf_drop'])
        for i, b in enumerate(BAND_ORDER):
            w.writerow([b, f'{band_drop[i]:.4f}', f'{band_conf_drop[i]:.4f}'])
    print(f'  Saved → {csv_path}')

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Band Importance — MS ResNet-50 (Occlusion Analysis)', fontsize=12, fontweight='bold')

    colors = ['#e74c3c' if d > 0 else '#95a5a6' for d in band_drop]
    axes[0].bar(BAND_ORDER, band_drop * 100, color=colors, edgecolor='white', linewidth=0.5)
    axes[0].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[0].set_xlabel('Spectral Band')
    axes[0].set_ylabel('Accuracy Drop (pp)')
    axes[0].set_title('Accuracy Drop When Band is Zeroed')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', alpha=0.3)

    colors2 = ['#e74c3c' if d > 0 else '#95a5a6' for d in band_conf_drop]
    axes[1].bar(BAND_ORDER, band_conf_drop * 100, color=colors2, edgecolor='white', linewidth=0.5)
    axes[1].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[1].set_xlabel('Spectral Band')
    axes[1].set_ylabel('Confidence Drop (pp)')
    axes[1].set_title('True-Class Confidence Drop When Band is Zeroed')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(axis='y', alpha=0.3)

    for ax in axes:
        ax.set_xticklabels([f'{b}\n({BAND_DESC[b]})' for b in BAND_ORDER], fontsize=7)

    plt.tight_layout()
    fig.savefig(OUT_DIR / 'band_importance.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved → {OUT_DIR}/band_importance.png')

    # Print summary
    print('\n  Band importance ranking (by accuracy drop):')
    ranked = sorted(zip(BAND_ORDER, band_drop), key=lambda x: -x[1])
    for b, d in ranked:
        bar = '█' * max(0, int(d * 500))
        print(f'    {b:4s} {BAND_DESC[b]:8s}  {d*100:+.2f}pp  {bar}')


# ══════════════════════════════════════════════════════════════════════════════
# 8. Spectral signatures per class
# ══════════════════════════════════════════════════════════════════════════════

print('\n── Computing spectral signatures per class ──')

if RASTERIO_OK:
    class_spectra = {i: [] for i in range(N_CLASSES)}

    for path, true_idx in tqdm(ms_items, desc='Spectra', leave=False):
        with rasterio.open(path) as src:
            raw = src.read().astype(np.float32)   # (13, 64, 64)
        mean_per_band = raw.mean(axis=(1, 2))      # (13,)
        class_spectra[true_idx].append(mean_per_band)

    # Color palette for 10 classes
    palette = plt.cm.tab10(np.linspace(0, 1, 10))

    fig, ax = plt.subplots(figsize=(12, 6))
    for idx in range(N_CLASSES):
        if not class_spectra[idx]:
            continue
        arr  = np.stack(class_spectra[idx])   # (N, 13)
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)
        x    = np.arange(13)
        ax.plot(x, mean, marker='o', markersize=4, label=CLASS_NAMES[idx],
                color=palette[idx], linewidth=1.5)
        ax.fill_between(x, mean - std, mean + std, alpha=0.08, color=palette[idx])

    ax.set_xticks(range(13))
    ax.set_xticklabels([f'{b}\n({BAND_DESC[b]})' for b in BAND_ORDER], fontsize=7)
    ax.set_xlabel('Spectral Band')
    ax.set_ylabel('Mean Raw DN Value')
    ax.set_title('Mean Spectral Signatures by Land-Use Class (EuroSAT Sentinel-2)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, ncol=2, loc='upper left')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT_DIR / 'spectral_signatures.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved → {OUT_DIR}/spectral_signatures.png')


# ══════════════════════════════════════════════════════════════════════════════
# 9. Summary
# ══════════════════════════════════════════════════════════════════════════════

print('\n' + '='*60)
print('EXPLAINABILITY OUTPUTS COMPLETE')
print('='*60)
outputs = list(OUT_DIR.glob('*.png')) + list(OUT_DIR.glob('*.csv'))
for f in sorted(outputs):
    size = f.stat().st_size // 1024
    print(f'  {f.name:40s}  {size:4d} KB')
print(f'\nAll saved to: {OUT_DIR.resolve()}')