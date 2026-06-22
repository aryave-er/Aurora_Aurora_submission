"""
utils.py — Preprocessing utilities, spectral indices, data augmentation, and metrics.
"""

from __future__ import annotations
import pandas as pd
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

# ──────────────────────────────────────────────────────────────────────────────
# Band Statistics — Sentinel-2 / EuroSAT Training Set
# ──────────────────────────────────────────────────────────────────────────────

BAND_STATS: Dict[str, Dict[str, float]] = {
    "B1":  {"wavelength_nm": 443,  "resolution_m": 60,  "p0": 1013.0,  "p98": 1370.19, "mean": 1354.9,  "std": 65.1,   "description": "Coastal aerosols, water column correction"},
    "B2":  {"wavelength_nm": 490,  "resolution_m": 10,  "p0": 676.0,   "p98": 1184.38, "mean": 1117.2,  "std": 154.0,  "description": "Bathymetric mapping, chlorophyll absorption"},
    "B3":  {"wavelength_nm": 560,  "resolution_m": 10,  "p0": 448.0,   "p98": 1120.77, "mean": 1040.5,  "std": 162.2,  "description": "Vegetation vigor, green peak reflectance"},
    "B4":  {"wavelength_nm": 665,  "resolution_m": 10,  "p0": 247.0,   "p98": 1136.26, "mean": 940.0,   "std": 209.8,  "description": "Chlorophyll absorption, vegetation mapping"},
    "B5":  {"wavelength_nm": 705,  "resolution_m": 20,  "p0": 269.0,   "p98": 1263.74, "mean": 1141.0,  "std": 208.8,  "description": "Red Edge 1: crop stress assessment"},
    "B6":  {"wavelength_nm": 740,  "resolution_m": 20,  "p0": 253.0,   "p98": 1645.40, "mean": 1533.8,  "std": 290.5,  "description": "Red Edge 2: vegetation state analysis"},
    "B7":  {"wavelength_nm": 783,  "resolution_m": 20,  "p0": 243.0,   "p98": 1846.87, "mean": 1704.3,  "std": 327.3,  "description": "Red Edge 3: leaf area index (LAI) estimation"},
    "B8":  {"wavelength_nm": 842,  "resolution_m": 10,  "p0": 189.0,   "p98": 1762.60, "mean": 1560.2,  "std": 372.7,  "description": "Broad NIR: biomass density"},
    "B8A": {"wavelength_nm": 865,  "resolution_m": 20,  "p0": 61.0,    "p98": 1972.62, "mean": 1763.2,  "std": 350.6,  "description": "Narrow NIR: leaf canopy structural detail"},
    "B9":  {"wavelength_nm": 940,  "resolution_m": 60,  "p0": 4.0,     "p98": 582.73,  "mean": 396.8,   "std": 145.5,  "description": "Water vapor correction, atmospheric filtering"},
    "B10": {"wavelength_nm": 1375, "resolution_m": 60,  "p0": 0.0,     "p98": 30.0,    "mean": 12.0,    "std": 4.9,    "description": "Cirrus cloud detection (highly skewed, near-zero for most scenes)"},
    "B11": {"wavelength_nm": 1610, "resolution_m": 20,  "p0": 11.0,    "p98": 1812.0,  "mean": 1374.4,  "std": 380.2,  "description": "SWIR 1: vegetation water content"},
    "B12": {"wavelength_nm": 2190, "resolution_m": 20,  "p0": 186.0,   "p98": 1124.0,  "mean": 847.5,   "std": 268.6,  "description": "SWIR 2: soil mineralogy, burn severity"},
}

BAND_NAMES = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
BAND_WAVELENGTHS = [BAND_STATS[b]["wavelength_nm"] for b in BAND_NAMES]

# Bands suitable for LOD log-normalization (highly right-skewed distributions)
LOD_BAND_SET = {"B1", "B9", "B10"}


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing Functions
# ──────────────────────────────────────────────────────────────────────────────

def quantile_clip_bands(
    array: np.ndarray,
    band_names: List[str] = BAND_NAMES,
) -> np.ndarray:
    """
    Clip each band using the 0th and 98th percentile statistics from the training set.
    """
    result = array.copy()
    for i, band in enumerate(band_names):
        if band not in BAND_STATS:
            warnings.warn(f"Band {band!r} not in BAND_STATS. Skipping clip.", stacklevel=2)
            continue
        stats = BAND_STATS[band]
        p0, p98 = stats["p0"], stats["p98"]
        lo, hi = min(p0, p98), max(p0, p98)
        result[i] = np.clip(array[i], lo, hi)
    return result


def log_normalize(x: np.ndarray, band_name: str) -> np.ndarray:
    """
    LOD(x) = ln(1 + x - x_min) / ln(1 + x_max - x_min)
    Maps values in [x_min, x_max] → [0, 1] using log scaling.
    """
    stats = BAND_STATS[band_name]
    p0, p98 = min(stats["p0"], stats["p98"]), max(stats["p0"], stats["p98"])
    numerator   = np.log1p(np.maximum(x - p0, 0.0))
    denominator = np.log1p(p98 - p0) + 1e-8
    return numerator / denominator


def zscore_normalize(x: np.ndarray, band_name: str) -> np.ndarray:
    """Z-score normalization using training set mean and std."""
    stats = BAND_STATS[band_name]
    return (x - stats["mean"]) / (stats["std"] + 1e-8)


def preprocess_ms_patch(
    raw: np.ndarray,
    band_names: List[str] = BAND_NAMES,
    append_indices: bool = False,
) -> np.ndarray:
    """
    Full preprocessing pipeline for a single multispectral patch.
    Steps: quantile clip → LOD (skewed bands) → Z-score → optional indices.
    """
    clipped = quantile_clip_bands(raw, band_names)
    processed = np.zeros_like(clipped, dtype=np.float32)

    for i, band in enumerate(band_names):
        if band in LOD_BAND_SET:
            processed[i] = log_normalize(clipped[i], band)
        else:
            processed[i] = zscore_normalize(clipped[i], band)

    if append_indices:
        indices = compute_spectral_indices(processed, band_names)
        processed = np.concatenate([processed, indices], axis=0)

    return processed


# ──────────────────────────────────────────────────────────────────────────────
# Biophysical Spectral Indices
# ──────────────────────────────────────────────────────────────────────────────

def _get_band(array: np.ndarray, band: str, band_names: List[str]) -> np.ndarray:
    """Retrieve a band slice from a (C, H, W) array by name."""
    try:
        idx = band_names.index(band)
        return array[idx].astype(np.float32)
    except ValueError:
        raise ValueError(f"Band {band!r} not found in band_names: {band_names}")


def compute_spectral_indices(
    array: np.ndarray,
    band_names: List[str] = BAND_NAMES,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Compute 9 biophysical spectral indices from a (C, H, W) array.
    Returns (9, H, W) float32 array.

    NDVI, EVI2, NDRE, NDMI, NDWI, MNDWI, NDBI, NDSI, NBR
    """
    def g(b): return _get_band(array, b, band_names)

    B3  = g("B3")
    B4  = g("B4")
    B5  = g("B5")
    B8  = g("B8")
    B8A = g("B8A")
    B11 = g("B11")
    B12 = g("B12")

    ndvi  = (B8 - B4)  / (B8 + B4  + eps)
    evi2  = 2.5 * (B8 - B4) / (B8 + 2.4 * B4 + 1.0 + eps)
    ndre  = (B8A - B5) / (B8A + B5 + eps)
    ndmi  = (B8 - B11) / (B8 + B11 + eps)
    ndwi  = (B3 - B8)  / (B3 + B8  + eps)
    mndwi = (B3 - B11) / (B3 + B11 + eps)
    ndbi  = (B11 - B8) / (B11 + B8 + eps)
    ndsi  = (B3 - B11) / (B3 + B11 + eps)
    nbr   = (B8 - B12) / (B8 + B12 + eps)

    return np.stack([ndvi, evi2, ndre, ndmi, ndwi, mndwi, ndbi, ndsi, nbr], axis=0).astype(np.float32)


INDEX_NAMES = ["NDVI", "EVI2", "NDRE", "NDMI", "NDWI", "MNDWI", "NDBI", "NDSI", "NBR"]


def route_pipeline(
    predicted_class: str,
    ms_patch: np.ndarray,
    band_names: List[str] = BAND_NAMES,
) -> Dict[str, float]:
    """Automated biophysical pipeline routing based on predicted land-cover class."""
    all_indices = compute_spectral_indices(ms_patch, band_names)
    idx_map = {name: all_indices[i] for i, name in enumerate(INDEX_NAMES)}

    routing = {
        "AnnualCrop":            ["NDVI", "NDRE"],
        "PermanentCrop":         ["NDVI", "NDRE", "EVI2"],
        "Forest":                ["NDMI", "NDVI", "NBR"],
        "HerbaceousVegetation":  ["NDVI", "EVI2", "NDRE"],
        "Pasture":               ["NDMI", "NDWI", "NDVI"],
        "Residential":           ["NDBI"],
        "Industrial":            ["NDBI", "NBR"],
        "River":                 ["NDWI", "MNDWI"],
        "SeaLake":               ["NDWI", "MNDWI", "NDSI"],
        "Highway":               ["NDBI", "NBR"],
    }

    target_indices = routing.get(predicted_class, INDEX_NAMES)
    return {name: float(idx_map[name].mean()) for name in target_indices if name in idx_map}


# ──────────────────────────────────────────────────────────────────────────────
# Spatial Leakage Coefficient (module-level convenience function)
# ──────────────────────────────────────────────────────────────────────────────

def compute_slc(acc_random: float, acc_spatial: float) -> float:
    """
    Spatial Leakage Coefficient: measures accuracy inflation from random splits.
    SLC = (acc_random - acc_spatial) / acc_random
    Higher → more leakage. Reference on EuroSAT ≈ 0.152.
    """
    return (acc_random - acc_spatial) / (acc_random + 1e-8)


# ──────────────────────────────────────────────────────────────────────────────
# CutMix (numpy version for direct array usage)
# ──────────────────────────────────────────────────────────────────────────────

class CutMixAugmentation:
    """CutMix data augmentation for multispectral image tensors (numpy version)."""

    def __init__(self, alpha: float = 1.0, prob: float = 0.5, num_classes: int = 10) -> None:
        self.alpha = alpha
        self.prob = prob
        self.num_classes = num_classes

    def __call__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply CutMix to a batch.
        images: (B, C, H, W), labels: (B,) integer
        Returns: mixed_images (B, C, H, W), soft_labels (B, num_classes)
        """
        B, C, H, W = images.shape
        one_hot = np.eye(self.num_classes)[labels]  # (B, num_classes)

        if np.random.rand() > self.prob:
            return images.copy(), one_hot

        lam = np.random.beta(self.alpha, self.alpha)
        rand_idx = np.random.permutation(B)
        shuffled_images = images[rand_idx]
        shuffled_one_hot = one_hot[rand_idx]

        cut_h = int(H * np.sqrt(1.0 - lam))
        cut_w = int(W * np.sqrt(1.0 - lam))

        cx = np.random.randint(W)
        cy = np.random.randint(H)
        x1 = max(cx - cut_w // 2, 0)
        x2 = min(cx + cut_w // 2, W)
        y1 = max(cy - cut_h // 2, 0)
        y2 = min(cy + cut_h // 2, H)

        mixed = images.copy()
        mixed[:, :, y1:y2, x1:x2] = shuffled_images[:, :, y1:y2, x1:x2]
        lam_actual = 1.0 - ((x2 - x1) * (y2 - y1)) / (H * W)

        soft_labels = lam_actual * one_hot + (1.0 - lam_actual) * shuffled_one_hot
        return mixed, soft_labels


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def topk_accuracy(outputs: torch.Tensor, targets: torch.Tensor, k: int = 1) -> float:
    """Top-K accuracy."""
    _, pred = outputs.topk(k, dim=1, largest=True, sorted=True)
    correct = pred.eq(targets.view(-1, 1).expand_as(pred))
    return correct.any(dim=1).float().mean().item()


def per_class_accuracy(
    preds: np.ndarray,
    targets: np.ndarray,
    class_names: List[str],
) -> Dict[str, float]:
    """Compute per-class accuracy."""
    result = {}
    for i, name in enumerate(class_names):
        mask = targets == i
        if mask.sum() == 0:
            result[name] = float("nan")
        else:
            result[name] = float((preds[mask] == i).mean())
    return result


def confidence_calibration_stats(
    probs: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, float]:
    """
    Compute confidence calibration statistics.
    Returns mean confidence for correct and incorrect predictions, plus the gap.
    """
    max_probs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct_mask = preds == targets

    mean_correct   = float(max_probs[correct_mask].mean())   if correct_mask.any()  else 0.0
    mean_incorrect = float(max_probs[~correct_mask].mean())  if (~correct_mask).any() else 0.0

    return {
        "mean_conf_correct":   mean_correct,
        "mean_conf_incorrect": mean_incorrect,
        "confidence_gap":      mean_correct - mean_incorrect,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Visualization Utilities
# ──────────────────────────────────────────────────────────────────────────────

def make_false_color(
    ms_patch: np.ndarray,
    band_names: List[str] = BAND_NAMES,
    mode: str = "nir_swir_red",
) -> np.ndarray:
    """
    Create a false-color composite for multispectral visualization.
    Modes: 'nir_swir_red', 'red_edge', 'swir_nir_green', 'natural'
    Returns (H, W, 3) uint8 array.
    """
    mode_map = {
        "nir_swir_red":   ["B8", "B11", "B4"],
        "red_edge":       ["B7", "B5", "B4"],
        "swir_nir_green": ["B11", "B8", "B3"],
        "natural":        ["B4", "B3", "B2"],
    }
    if mode not in mode_map:
        raise ValueError(f"Unknown mode {mode!r}. Choose from {list(mode_map)}")

    channels = []
    for band in mode_map[mode]:
        ch = _get_band(ms_patch, band, band_names).astype(np.float32)
        lo, hi = np.percentile(ch, 2), np.percentile(ch, 98)
        ch = np.clip((ch - lo) / (hi - lo + 1e-8), 0, 1)
        channels.append((ch * 255).astype(np.uint8))

    return np.stack(channels, axis=-1)


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    title: str = "Training Curves",
    save_path: Optional[str] = None,
) -> None:
    """Plot training and validation loss + accuracy curves."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.style as mstyle
        mstyle.use("dark_background")
    except ImportError:
        warnings.warn("matplotlib not installed. Cannot plot curves.", stacklevel=2)
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0d1117")
    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, "b-o", markersize=4, label="Train Loss", linewidth=2)
    ax1.plot(epochs, val_losses,   "r-s", markersize=4, label="Val Loss",   linewidth=2)
    ax1.set_title("Loss", color="white", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch", color="white")
    ax1.set_ylabel("Loss",  color="white")
    ax1.legend(facecolor="#1a1a2e")
    ax1.set_facecolor("#0d1117")
    ax1.tick_params(colors="white")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#30363d")

    ax2.plot(epochs, [a * 100 for a in train_accs], "b-o", markersize=4, label="Train Acc", linewidth=2)
    ax2.plot(epochs, [a * 100 for a in val_accs],   "r-s", markersize=4, label="Val Acc",   linewidth=2)
    ax2.set_title("Accuracy (%)", color="white", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch", color="white")
    ax2.set_ylabel("Accuracy (%)", color="white")
    ax2.legend(facecolor="#1a1a2e")
    ax2.set_facecolor("#0d1117")
    ax2.tick_params(colors="white")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#30363d")

    fig.suptitle(title, color="white", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
        print(f"Saved training curves to {save_path}")
    plt.show()


def confusion_matrix_fig(
    preds: np.ndarray,
    targets: np.ndarray,
    class_names: List[str],
    normalize: bool = True,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
) -> None:
    """Plot a styled confusion matrix using seaborn."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
    except ImportError:
        warnings.warn("matplotlib/seaborn/sklearn required for confusion matrix.", stacklevel=2)
        return

    cm = confusion_matrix(targets, preds, labels=list(range(len(class_names))))
    if normalize:
        cm = cm.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.where(row_sums > 0, cm / row_sums, 0.0)
        fmt, vmax = ".2f", 1.0
    else:
        fmt, vmax = "d", None

    fig, ax = plt.subplots(figsize=(12, 10), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="viridis",
        vmin=0,
        vmax=vmax,
        ax=ax,
        linewidths=0.5,
        linecolor="#30363d",
    )
    ax.set_xlabel("Predicted", color="white", fontsize=12)
    ax.set_ylabel("True",      color="white", fontsize=12)
    ax.set_title(title,        color="white", fontsize=15, fontweight="bold")
    ax.tick_params(colors="white", labelsize=9)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
        print(f"Saved confusion matrix to {save_path}")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Training Loop Helpers
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scheduler=None,
    scaler=None,
    soft_labels: bool = False,
) -> Tuple[float, float]:
    """
    Run one training epoch with optional mixed precision (CUDA only; MPS/CPU
    run without a scaler).

    Args:
        soft_labels: If True, labels are float tensors (e.g. from CutMix).
    Returns:
        (mean_loss, mean_accuracy)
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # AMP is only meaningful on CUDA; on MPS/CPU we skip the scaler
    use_amp = (scaler is not None) and (device.type == "cuda")

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

        if soft_labels and labels.dim() > 1:
            hard_labels = labels.argmax(dim=1)
        else:
            hard_labels = labels
        preds = outputs.argmax(dim=1)
        total_correct += (preds == hard_labels).sum().item()

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate model on a DataLoader. Handles both hard integer labels and
    soft one-hot labels (e.g. from CutMix val sets).
    Returns: (mean_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

        preds = outputs.argmax(dim=1)
        # Handle both hard integer labels and soft one-hot labels
        hard_labels = labels.argmax(dim=1) if labels.dim() > 1 else labels
        total_correct += (preds == hard_labels).sum().item()

    return total_loss / total_samples, total_correct / total_samples


# ──────────────────────────────────────────────────────────────────────────────
# Prediction generation for competition submission
# ──────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def generate_test_predictions(
    model: nn.Module,
    test_loader,
    device: torch.device,
    output_path: str,
    class_names: List[str],
) -> "pd.DataFrame":
    """
    Run inference on the unlabeled test flat directory and save competition CSV.

    The competition requires:
        rgb_predictions.csv  →  img_id, predicted_label
        ms_predictions.csv   →  img_id, predicted_label

    Args:
        model:       Trained model in eval mode.
        test_loader: DataLoader from EuroSATTestDataset (returns tensor, img_id).
        device:      Torch device.
        output_path: Where to save the CSV.
        class_names: List of class name strings.

    Returns:
        pd.DataFrame with img_id and predicted_label columns.
    """
    

    model.eval()
    model.to(device)

    all_img_ids = []
    all_preds = []
    all_probs = []

    for images, img_ids in test_loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        all_img_ids.extend(img_ids if isinstance(img_ids, list) else list(img_ids))
        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())

    df = pd.DataFrame({
        "img_id": all_img_ids,
        "predicted_label": all_preds,
        "predicted_class": [class_names[p] for p in all_preds],
        "confidence": [max(p) for p in all_probs],
        **{f"prob_{cls}": [p[i] for p in all_probs] for i, cls in enumerate(class_names)},
    })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df[["img_id", "predicted_label"]].to_csv(output_path, index=False)
    print(f"✅ Saved {len(df)} predictions → {output_path}")
    return df