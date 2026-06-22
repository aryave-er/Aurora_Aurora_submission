# src/__init__.py

from .dataset import EuroSATRGBDataset, EuroSATMSDataset, SpatialKFoldSplitter
from .models import (
    construct_multispectral_resnet50,
    construct_ms_efficientnet,
    BalancedAttentionNet,
    get_swin_transformer,
    get_efficientnet_b0,
)
from .utils import (
    BAND_STATS,
    quantile_clip_bands,
    log_normalize,
    zscore_normalize,
    compute_spectral_indices,
    CutMixAugmentation,
    compute_slc,
)

__version__ = "1.0.0"
__project__ = "Aurora — Geo Snap Paradigm"