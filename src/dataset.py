"""
dataset.py — PyTorch Dataset classes for EuroSAT RGB and Multispectral imagery,
plus SpatialKFoldSplitter for geographically independent cross-validation.

Supports flat class-dir layout, competition train/val/test_flat layout,
and CSV-based loading via train.csv / validation.csv.
"""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    warnings.warn(
        "rasterio not installed. EuroSATMSDataset will not be available. "
        "Install with: pip install rasterio",
        ImportWarning,
        stacklevel=2,
    )

try:
    from .utils import BAND_STATS
except ImportError:
    from utils import BAND_STATS

# ──────────────────────────────────────────────────────────────────────────────
# EuroSAT Class Mapping
# ──────────────────────────────────────────────────────────────────────────────

EUROSAT_CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

CLASS_TO_IDX: Dict[str, int] = {cls: i for i, cls in enumerate(EUROSAT_CLASSES)}
IDX_TO_CLASS: Dict[int, str] = {i: cls for cls, i in CLASS_TO_IDX.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Band Statistics (Training set 0th / 98th percentile for Sentinel-2)
# Derived from utils.BAND_STATS so there is a single source of truth for
# clip/normalization values shared across dataset.py and utils.py.
# ──────────────────────────────────────────────────────────────────────────────

BAND_ORDER = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]

BAND_CLIP_STATS = {
    band: (BAND_STATS[band]["p0"], BAND_STATS[band]["p98"],
           BAND_STATS[band]["mean"], BAND_STATS[band]["std"])
    for band in BAND_ORDER
}

LOD_BANDS = set()

# Fallback RGB stats, used only if outputs/rgb_means.npy / rgb_stds.npy
# (written by train_swin.py) are not yet present.
_FALLBACK_RGB_MEAN = [0.3444, 0.3803, 0.4078]
_FALLBACK_RGB_STD  = [0.2029, 0.1366, 0.1153]


def _load_rgb_stats(stats_dir: str | Path = "outputs") -> Tuple[List[float], List[float]]:
    """Load RGB mean/std computed by train_swin.py, falling back to defaults."""
    stats_dir = Path(stats_dir)
    mean_path, std_path = stats_dir / "rgb_means.npy", stats_dir / "rgb_stds.npy"
    if mean_path.exists() and std_path.exists():
        return np.load(mean_path).tolist(), np.load(std_path).tolist()
    return _FALLBACK_RGB_MEAN, _FALLBACK_RGB_STD


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_file_list_from_dir(root: Path, extensions: Tuple[str, ...]) -> List[Tuple[str, int, str]]:
    """
    Traverse root/<ClassName>/<file> structure.
    Returns list of (filepath, label_idx, class_name).
    """
    entries = []
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        label = CLASS_TO_IDX.get(class_name)
        if label is None:
            warnings.warn(f"Unknown class directory: {class_name!r}. Skipping.", stacklevel=2)
            continue
        for f in sorted(class_dir.iterdir()):
            if f.suffix.lower() in extensions:
                entries.append((str(f), label, class_name))
    return entries


def _build_file_list_from_csv(csv_path: Path, data_root: Path, extensions: Tuple[str, ...]) -> List[Tuple[str, int, str]]:
    """
    Build file list from official train.csv / validation.csv.
    Expected CSV columns: filename, label  (or: filename, ClassName)
    Tries to resolve filepath as data_root/filename or data_root/<ClassName>/filename.
    """
    df = pd.read_csv(csv_path)

    # Normalize column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    # Detect label column
    label_col = None
    for candidate in ["label", "class", "classname", "class_name", "category"]:
        if candidate in df.columns:
            label_col = candidate
            break

    if label_col is None:
        raise ValueError(f"Cannot find label column in {csv_path}. Columns: {list(df.columns)}")

    # Detect filename column
    fname_col = None
    for candidate in ["filename", "file", "filepath", "path", "image"]:
        if candidate in df.columns:
            fname_col = candidate
            break

    if fname_col is None:
        raise ValueError(f"Cannot find filename column in {csv_path}. Columns: {list(df.columns)}")

    entries = []
    for _, row in df.iterrows():
        # Normalize Windows-style backslashes → forward slashes (cross-platform)
        fname = str(row[fname_col]).strip().replace('\\', '/')
        raw_label = row[label_col]

        # Resolve label to integer
        if isinstance(raw_label, str):
            class_name = raw_label.strip()
            label = CLASS_TO_IDX.get(class_name)
            if label is None:
                warnings.warn(f"Unknown class {class_name!r} in CSV. Skipping.", stacklevel=2)
                continue
        else:
            label = int(raw_label)
            class_name = IDX_TO_CLASS.get(label, "Unknown")

        # CSV paths may be "AnnualCrop\AnnualCrop_683.jpg" (Windows style)
        # Extract bare filename — actual files live in train/ClassName/bare
        bare = Path(fname).name  # e.g. "AnnualCrop_683.jpg"

        candidates = [
            data_root / "train" / class_name / bare,
            data_root / "val"   / class_name / bare,
            data_root / class_name / bare,
            data_root / "train" / bare,
            data_root / "val"   / bare,
            data_root / bare,
        ]
        found = None
        for c in candidates:
            if c.exists() and c.suffix.lower() in extensions:
                found = str(c)
                break

        if found is not None:
            entries.append((found, label, class_name))
        else:
            warnings.warn(f"File not found for {fname!r}. Skipping.", stacklevel=2)

    return entries


def build_file_list(
    root: str | Path,
    extensions: Tuple[str, ...],
    csv_path: Optional[str | Path] = None,
) -> List[Tuple[str, int, str]]:
    """
    Smart file list builder. If csv_path is given, uses CSV-based loading.
    Otherwise falls back to directory traversal.
    Also handles train/ and val/ subdirectories automatically.
    """
    root = Path(root)

    if csv_path is not None:
        return _build_file_list_from_csv(Path(csv_path), root, extensions)

    # Check if root has train/ and val/ subdirs (competition layout)
    has_split_dirs = (root / "train").exists() or (root / "val").exists()
    if has_split_dirs:
        entries = []
        for split in ["train", "val"]:
            split_dir = root / split
            if split_dir.exists():
                entries.extend(_build_file_list_from_dir(split_dir, extensions))
        if entries:
            return entries

    # Fall back to flat class-dir layout
    return _build_file_list_from_dir(root, extensions)


# ──────────────────────────────────────────────────────────────────────────────
# RGB Dataset
# ──────────────────────────────────────────────────────────────────────────────

class EuroSATRGBDataset(Dataset):
    """
    Loads EuroSAT RGB JPEG patches (64×64, 3 channels).

    Supports:
        - Flat class-dir layout: root/<ClassName>/<file.jpg>
        - Competition layout:    root/train/<ClassName>/<file.jpg>
        - CSV-based loading:     pass csv_path=train.csv

    Normalization stats are loaded from outputs/rgb_means.npy / rgb_stds.npy
    if present (written by train_swin.py), otherwise a fallback is used.

    ColorJitter is intentionally excluded from augmentation: it alters
    brightness/contrast in ways that change the physical relationship
    between bands, which is incorrect for satellite imagery. CutMix
    handles spatial regularization instead.
    """

    def __init__(
        self,
        root: str | Path,
        indices: Optional[List[int]] = None,
        transform: Optional[Callable] = None,
        is_train: bool = True,
        csv_path: Optional[str | Path] = None,
    ) -> None:
        self.root = Path(root)
        self.is_train = is_train
        self.mean, self.std = _load_rgb_stats()

        all_entries = build_file_list(self.root, (".jpg", ".jpeg", ".png"), csv_path)
        if not all_entries:
            raise FileNotFoundError(
                f"No image files found under {self.root}. "
                "Check your data layout or pass csv_path= for CSV-based loading."
            )

        if indices is not None:
            self.entries = [all_entries[i] for i in indices]
        else:
            self.entries = all_entries

        self.transform = transform or self._default_transform()

    def _default_transform(self) -> transforms.Compose:
        if self.is_train:
            return transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=360),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ])
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        filepath, label, _ = self.entries[idx]
        image = Image.open(filepath).convert("RGB")
        tensor = self.transform(image)
        return tensor, label

    @property
    def class_names(self) -> List[str]:
        return EUROSAT_CLASSES

    @property
    def targets(self) -> List[int]:
        return [e[1] for e in self.entries]

    @property
    def filepaths(self) -> List[str]:
        return [e[0] for e in self.entries]


# ──────────────────────────────────────────────────────────────────────────────
# Test Dataset (unlabeled flat directory)
# ──────────────────────────────────────────────────────────────────────────────

class EuroSATTestDataset(Dataset):
    """
    Loads unlabeled test images from a flat directory (EuroSAT_test_flat/).
    Returns (tensor, img_id) where img_id is the filename stem.

    This is what generates the final rgb_predictions.csv and ms_predictions.csv
    that the competition requires.

    Args:
        flat_dir:  Path to flat directory of unlabeled test images.
        is_ms:     If True, loads 13-band TIFF. If False, loads RGB JPEG.
        transform: Optional transform override.
    """

    def __init__(
        self,
        flat_dir: str | Path,
        is_ms: bool = False,
        transform: Optional[Callable] = None,
    ) -> None:
        self.flat_dir = Path(flat_dir)
        self.is_ms = is_ms
        self.rgb_mean, self.rgb_std = _load_rgb_stats()

        if is_ms:
            if not RASTERIO_AVAILABLE:
                raise ImportError("rasterio required for MS test loading. pip install rasterio")
            exts = {".tif", ".tiff"}
        else:
            exts = {".jpg", ".jpeg", ".png"}

        self.files = sorted([f for f in self.flat_dir.iterdir() if f.suffix.lower() in exts])

        if not self.files:
            raise FileNotFoundError(f"No {'TIFF' if is_ms else 'image'} files found under {self.flat_dir}")

        self.transform = transform or self._default_transform()

    def _default_transform(self) -> Callable:
        if self.is_ms:
            return lambda x: torch.from_numpy(x)  # preprocessing done in __getitem__
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.rgb_mean, std=self.rgb_std),
        ])

    def _preprocess_ms(self, raw: np.ndarray) -> np.ndarray:
        """Apply same preprocessing pipeline as EuroSATMSDataset."""
        processed = np.zeros_like(raw, dtype=np.float32)
        for i, band_name in enumerate(BAND_ORDER):
            band = raw[i].astype(np.float32)
            p0, p98, mean, std = BAND_CLIP_STATS[band_name]
            lo, hi = min(p0, p98), max(p0, p98)
            band = np.clip(band, lo, hi)
            if band_name in LOD_BANDS:
                numerator = np.log1p(np.maximum(band - lo, 0.0))
                denominator = np.log1p(hi - lo) + 1e-8
                band = numerator / denominator
            else:
                band = (band - mean) / (std + 1e-8)
            processed[i] = band
        return processed

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        f = self.files[idx]
        img_id = f.stem  # filename without extension → used as img_id in CSV

        if self.is_ms:
            with rasterio.open(f) as src:
                raw = src.read().astype(np.float32)
            if raw.shape[0] < 13:
                pad = np.zeros((13 - raw.shape[0], *raw.shape[1:]), dtype=np.float32)
                raw = np.concatenate([raw, pad], axis=0)
            raw = raw[:13]
            processed = self._preprocess_ms(raw)
            return torch.from_numpy(processed), img_id
        else:
            image = Image.open(f).convert("RGB")
            return self.transform(image), img_id

    @property
    def img_ids(self) -> List[str]:
        return [f.stem for f in self.files]


# ──────────────────────────────────────────────────────────────────────────────
# Multispectral Dataset
# ──────────────────────────────────────────────────────────────────────────────

class EuroSATMSDataset(Dataset):
    """
    Loads EuroSAT multispectral TIFF patches (64×64, 13 Sentinel-2 bands).

    Preprocessing pipeline:
        1. Raw 16-bit float read via rasterio  →  shape (13, H, W)
        2. Per-band quantile clipping (0th / 98th percentile)
        3. LOD log-normalization for B1, B9, B10 (skewed distributions)
        4. Z-score normalization for all other bands
        5. Optional: append biophysical index channels (NDVI, NDRE, NDBI, NDMI)

    Supports same layouts as EuroSATRGBDataset.
    """

    def __init__(
        self,
        root: str | Path,
        indices: Optional[List[int]] = None,
        is_train: bool = True,
        append_indices: bool = False,
        csv_path: Optional[str | Path] = None,
    ) -> None:
        if not RASTERIO_AVAILABLE:
            raise ImportError("rasterio is required for EuroSATMSDataset. pip install rasterio")

        self.root = Path(root)
        self.is_train = is_train
        self.append_indices = append_indices

        all_entries = build_file_list(self.root, (".tif", ".tiff"), csv_path)
        if not all_entries:
            raise FileNotFoundError(
                f"No TIFF files found under {self.root}. "
                "Check your data layout or pass csv_path= for CSV-based loading."
            )

        if indices is not None:
            self.entries = [all_entries[i] for i in indices]
        else:
            self.entries = all_entries

    # ------------------------------------------------------------------
    # Preprocessing utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _quantile_clip(band_data: np.ndarray, band_name: str) -> np.ndarray:
        p0, p98, _, _ = BAND_CLIP_STATS[band_name]
        lo, hi = min(p0, p98), max(p0, p98)
        return np.clip(band_data, lo, hi)

    @staticmethod
    def _lod_normalize(x: np.ndarray, band_name: str) -> np.ndarray:
        """Logarithmic normalization: LOD(x) = ln(1 + x - x_min) / ln(1 + x_max - x_min)"""
        p0, p98, _, _ = BAND_CLIP_STATS[band_name]
        lo, hi = min(p0, p98), max(p0, p98)
        numerator = np.log1p(np.maximum(x - lo, 0.0))
        denominator = np.log1p(hi - lo) + 1e-8
        return numerator / denominator

    @staticmethod
    def _zscore(x: np.ndarray, band_name: str) -> np.ndarray:
        _, _, mean, std = BAND_CLIP_STATS[band_name]
        return (x - mean) / (std + 1e-8)

    def _preprocess_bands(self, raw: np.ndarray) -> np.ndarray:
        """Full preprocessing pipeline for one patch. raw: (13, H, W) float32"""
        processed = np.zeros_like(raw, dtype=np.float32)
        for i, band_name in enumerate(BAND_ORDER):
            band = raw[i].astype(np.float32)
            band = self._quantile_clip(band, band_name)
            if band_name in LOD_BANDS:
                band = self._lod_normalize(band, band_name)
            else:
                band = self._zscore(band, band_name)
            processed[i] = band
        return processed

    @staticmethod
    def _compute_indices(bands: np.ndarray) -> np.ndarray:
        """
        Compute 4 biophysical indices from preprocessed band array.
        Returns (4, H, W) array: [NDVI, NDRE, NDBI, NDMI]
        """
        eps = 1e-8
        B4  = bands[3]   # Red
        B5  = bands[4]   # Red Edge 1
        B8  = bands[7]   # NIR
        B8A = bands[8]   # Narrow NIR
        B11 = bands[11]  # SWIR1

        ndvi = (B8 - B4)  / (B8 + B4  + eps)
        ndre = (B8A - B5) / (B8A + B5 + eps)
        ndbi = (B11 - B8) / (B11 + B8 + eps)
        ndmi = (B8 - B11) / (B8 + B11 + eps)

        return np.stack([ndvi, ndre, ndbi, ndmi], axis=0)

    def _augment(self, arr: np.ndarray) -> np.ndarray:
        """Apply rotationally-invariant spatial augmentations. arr: (C, H, W)"""
        if np.random.rand() < 0.5:
            arr = np.flip(arr, axis=2).copy()  # horizontal flip
        if np.random.rand() < 0.5:
            arr = np.flip(arr, axis=1).copy()  # vertical flip
        k = np.random.randint(0, 4)
        arr = np.rot90(arr, k=k, axes=(1, 2)).copy()
        return arr

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        filepath, label, _ = self.entries[idx]

        with rasterio.open(filepath) as src:
            raw = src.read().astype(np.float32)  # (bands, H, W)

        # Ensure exactly 13 bands
        if raw.shape[0] < 13:
            pad = np.zeros((13 - raw.shape[0], *raw.shape[1:]), dtype=np.float32)
            raw = np.concatenate([raw, pad], axis=0)
        raw = raw[:13]

        processed = self._preprocess_bands(raw)

        if self.append_indices:
            indices = self._compute_indices(processed)
            processed = np.concatenate([processed, indices], axis=0)  # (17, H, W)

        if self.is_train:
            processed = self._augment(processed)

        tensor = torch.from_numpy(processed)
        return tensor, label

    @property
    def class_names(self) -> List[str]:
        return EUROSAT_CLASSES

    @property
    def targets(self) -> List[int]:
        return [e[1] for e in self.entries]

    @property
    def filepaths(self) -> List[str]:
        return [e[0] for e in self.entries]


# ──────────────────────────────────────────────────────────────────────────────
# Spatial K-Fold Splitter
# ──────────────────────────────────────────────────────────────────────────────

class SpatialKFoldSplitter:
    """
    Geographically aware K-Fold cross-validation to prevent spatial autocorrelation leakage.

    Strategy:
        1. Extract (lat, lon) centroid coordinates for each image patch.
        2. Cluster centroids into K geographic blocks via K-Means.
        3. Use GroupKFold with cluster IDs as group labels.

    This ensures training and validation sets come from spatially disjoint regions.
    """

    def __init__(self, n_splits: int = 5, random_state: int = 42) -> None:
        self.n_splits = n_splits
        self.random_state = random_state
        self._kmeans: Optional[KMeans] = None
        self._cluster_labels: Optional[np.ndarray] = None

    def fit(self, coords: np.ndarray) -> "SpatialKFoldSplitter":
        if coords.shape[0] < self.n_splits:
            raise ValueError(
                f"Cannot create {self.n_splits} folds with only {coords.shape[0]} samples."
            )
        self._kmeans = KMeans(
            n_clusters=self.n_splits,
            random_state=self.random_state,
            n_init="auto",
        )
        self._cluster_labels = self._kmeans.fit_predict(coords)
        return self

    def split(self, X: Any, y: Optional[np.ndarray] = None):
        if self._cluster_labels is None:
            raise RuntimeError("Call .fit(coords) before .split().")

        gkf = GroupKFold(n_splits=self.n_splits)
        n = len(self._cluster_labels)
        dummy_X = np.arange(n).reshape(-1, 1)

        for train_idx, val_idx in gkf.split(dummy_X, y, groups=self._cluster_labels):
            yield train_idx, val_idx

    @property
    def cluster_labels(self) -> Optional[np.ndarray]:
        return self._cluster_labels

    @property
    def cluster_centers(self) -> Optional[np.ndarray]:
        return self._kmeans.cluster_centers_ if self._kmeans else None

    def compute_slc(self, acc_random: float, acc_spatial: float) -> float:
        """SLC = (acc_random - acc_spatial) / acc_random"""
        return (acc_random - acc_spatial) / (acc_random + 1e-8)


# ──────────────────────────────────────────────────────────────────────────────
# CutMix Collator
# ──────────────────────────────────────────────────────────────────────────────

class CutMixCollator:
    """
    Batch-level CutMix augmentation for DataLoader.
    Mixes spatial regions between pairs of images and blends their labels.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        prob: float = 0.5,
        num_classes: int = 10,
    ) -> None:
        self.alpha = alpha
        self.prob = prob
        self.num_classes = num_classes

    def __call__(
        self, batch: List[Tuple[torch.Tensor, int]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        images = torch.stack([b[0] for b in batch])
        labels = torch.tensor([b[1] for b in batch], dtype=torch.long)

        if np.random.rand() > self.prob:
            one_hot = torch.zeros(len(labels), self.num_classes)
            one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
            return images, one_hot

        lam = np.random.beta(self.alpha, self.alpha)
        rand_idx = torch.randperm(images.size(0))
        shuffled_images = images[rand_idx]
        shuffled_labels = labels[rand_idx]

        _, _, H, W = images.shape
        cut_h = int(H * np.sqrt(1.0 - lam))
        cut_w = int(W * np.sqrt(1.0 - lam))

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        x1 = max(cx - cut_w // 2, 0)
        x2 = min(cx + cut_w // 2, W)
        y1 = max(cy - cut_h // 2, 0)
        y2 = min(cy + cut_h // 2, H)

        mixed = images.clone()
        mixed[:, :, y1:y2, x1:x2] = shuffled_images[:, :, y1:y2, x1:x2]

        lam = 1.0 - ((x2 - x1) * (y2 - y1)) / (H * W)

        one_hot_a = torch.zeros(len(labels), self.num_classes).scatter_(
            1, labels.unsqueeze(1), 1.0
        )
        one_hot_b = torch.zeros(len(shuffled_labels), self.num_classes).scatter_(
            1, shuffled_labels.unsqueeze(1), 1.0
        )
        mixed_labels = lam * one_hot_a + (1.0 - lam) * one_hot_b

        return mixed, mixed_labels


# ──────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ──────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    rgb_root: str | Path,
    ms_root: str | Path,
    train_indices: List[int],
    val_indices: List[int],
    batch_size: int = 32,
    num_workers: int = 4,
    use_cutmix: bool = True,
    append_spectral_indices: bool = False,
    rgb_csv: Optional[str | Path] = None,
    ms_csv: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    Build train/val DataLoaders for RGB and MS modalities.
    Accepts optional csv_path for official split CSVs.
    Returns dict with keys: 'rgb_train', 'rgb_val', 'ms_train', 'ms_val'
    """
    from torch.utils.data import DataLoader

    rgb_train = EuroSATRGBDataset(rgb_root, indices=train_indices, is_train=True, csv_path=rgb_csv)
    rgb_val   = EuroSATRGBDataset(rgb_root, indices=val_indices,   is_train=False, csv_path=rgb_csv)
    ms_train  = EuroSATMSDataset(ms_root, indices=train_indices, is_train=True,
                                  append_indices=append_spectral_indices, csv_path=ms_csv)
    ms_val    = EuroSATMSDataset(ms_root, indices=val_indices, is_train=False,
                                  append_indices=append_spectral_indices, csv_path=ms_csv)

    collator = CutMixCollator(alpha=1.0, prob=0.5, num_classes=10) if use_cutmix else None

    return {
        "rgb_train": DataLoader(
            rgb_train, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, collate_fn=collator, pin_memory=True,
        ),
        "rgb_val": DataLoader(
            rgb_val, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        ),
        "ms_train": DataLoader(
            ms_train, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
        ),
        "ms_val": DataLoader(
            ms_val, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        ),
    }