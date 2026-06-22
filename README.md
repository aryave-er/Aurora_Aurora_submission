# Aurora — Geo Snap Paradigm
### Land-Use Classification & Explainability from Space
**GeoSnap Competition — SOI × Cosmosoc**

---

## Results

| Model | Modality | Architecture | Val Accuracy |
|-------|----------|-------------|-------------|
| Swin-T | RGB (3 bands) | Transformer | 96.58% |
| ResNet-50 MS | 13-band Sentinel-2 | CNN + weight surgery | **99.37%** |

---

## Repository Structure

```
Aurora_Aurora_submission/
├── Data/                              ← dataset (not committed, see below)
├── notebooks/
│   ├── 01_spatial_validation.ipynb   ← domain shift analysis & split validation
│   ├── 02_multispectral_models.ipynb ← MS model training & evaluation
│   ├── 03_explainability_analysis.ipynb ← Task 2: XAI outputs & analysis
│   └── 04_environmental_mapping.ipynb   ← Task 3: spectral indices & insights
├── src/
│   ├── dataset.py                    ← class mappings & dataset utilities
│   ├── models.py                     ← Swin-T, ResNet-50 MS, architectures
│   ├── utils.py                      ← normalization, spectral indices, metrics
│   ├── train_swin.py                 ← RGB Swin-T training script
│   └── train_ms.py                   ← MS ResNet-50 training script
├── outputs/
│   ├── rgb_predictions.csv           ← Task 1 RGB predictions (4050 images)
│   ├── ms_predictions.csv            ← Task 1 MS predictions (4050 images)
│   ├── swin_transformer_rgb_best.pth ← Swin-T checkpoint (96.58%)
│   ├── resnet50_ms_surgery_best.pth  ← ResNet-50 MS checkpoint (99.37%)
│   └── xai/                          ← Task 2 explainability outputs
│       ├── gradcam_correct.png
│       ├── gradcam_nearmiss.png
│       ├── band_importance.png
│       ├── band_importance.csv
│       ├── spectral_signatures.png
│       ├── confusion_matrix.png
│       └── calibration.png
├── figures/                          ← figures referenced in report
├── explainability.py                 ← standalone XAI generation script
├── report.md                         ← technical report
├── requirements.txt
└── README.md
```

---

## Setup

### Requirements

Python 3.12+ recommended. Install dependencies:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install timm rasterio numpy pandas matplotlib tqdm pillow
```

For Apple Silicon (MPS acceleration):
```bash
pip install torch torchvision torchaudio
pip install timm rasterio numpy pandas matplotlib tqdm pillow
```

Or install all at once from requirements.txt:
```bash
pip install -r requirements.txt
```

### Dataset Setup

Download the EuroSAT dataset and place it under `Data/`:

```
Data/
├── EuroSAT/                    ← RGB JPEG images
│   ├── train/
│   └── val/
├── EuroSATallBands/            ← 13-band GeoTIFF images
│   ├── train/
│   └── val/
├── EuroSAT_test_flat/          ← unlabeled RGB test images
├── EuroSATallBands_test_flat/  ← unlabeled MS test images
├── label_map.json
├── train.csv
└── validation.csv
```

---

## Reproducing Results

### Train RGB Model (Swin-T)

```bash
cd Aurora_Aurora_submission
python3 src/train_swin.py
# Saves checkpoint → outputs/swin_transformer_rgb_best.pth
# Expected: ~96.58% val accuracy after 40 epochs (~2-3 hours on M4)
```

### Train MS Model (ResNet-50)

```bash
python3 src/train_ms.py
# Saves checkpoint → outputs/resnet50_ms_surgery_best.pth
# Expected: ~99.37% val accuracy after 35 epochs (~3-4 hours on M4)
# Note: first run computes global normalization stats (~5 min) and saves to outputs/
```

### Generate Predictions (Task 1)

```bash
python3 -c "
import sys, torch, numpy as np, csv
sys.path.insert(0, 'src')
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from models import get_swin_transformer, construct_multispectral_resnet50
import rasterio

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
MEAN = np.load('outputs/rgb_means.npy').tolist()
STD  = np.load('outputs/rgb_stds.npy').tolist()
tf   = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

rgb_model = get_swin_transformer(num_classes=10, pretrained=False).to(DEVICE)
rgb_model.load_state_dict(torch.load('outputs/swin_transformer_rgb_best.pth', map_location=DEVICE))
rgb_model.eval()

rows = []
for f in tqdm(sorted(Path('Data/EuroSAT_test_flat').iterdir()), desc='RGB'):
    if f.suffix.lower() not in ('.jpg','.jpeg','.png'): continue
    t = tf(Image.open(f).convert('RGB')).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = F.softmax(rgb_model(t), dim=1).argmax(1).item()
    rows.append((f.name, pred))

with open('outputs/rgb_predictions.csv', 'w', newline='') as out:
    w = csv.writer(out); w.writerow(['img_id','predicted_label']); w.writerows(rows)
print(f'Done: {len(rows)} RGB predictions')
"
```

Pre-generated predictions are already committed at `outputs/rgb_predictions.csv` and `outputs/ms_predictions.csv`.

### Generate Explainability Outputs (Task 2)

```bash
python3 explainability.py
# Generates all XAI outputs → outputs/xai/
# Runtime: ~5-8 minutes on M4 (band importance is slowest step)
```

Then open `notebooks/03_explainability_analysis.ipynb` and run all cells to view the analysis.

### Environmental Insights (Task 3)

Open `notebooks/04_environmental_mapping.ipynb` and run all cells.

---

## Key Design Decisions

**Why pool train+val directories?**
The official split has a severe phenological domain shift — AnnualCrop NIR means differ 2× between train/ and val/ due to different crop growth stages at acquisition time. Training on the official split gave 10% val accuracy (worse than random). Pooling all 22,950 files and re-splitting 80/20 with seed=42 resolves this.

**Why global normalization?**
Per-file normalization forces each band to N(0,1) independently, erasing inter-band ratios. Spectral indices like NDVI = (B8−B4)/(B8+B4) become meaningless. Global stats preserve physical spectral signatures.

**Why Kaiming init for non-optical MS channels?**
Initializing with mean RGB weights makes all 10 non-optical channels identical at t=0, preventing band-specific learning. Kaiming normal breaks this symmetry.

**Why no ColorJitter?**
ColorJitter alters brightness/contrast, changing physical band ratios and making augmented images spectrally implausible.

**Why occlusion over SHAP for band importance?**
SHAP DeepExplainer is unstable with BatchNorm under MPS and requires large background datasets. Occlusion is model-agnostic and produces physically interpretable results directly mappable to Sentinel-2 band properties.

---

## Hardware Notes

Trained on Apple M4 (MPS backend). If running on CUDA:
- Remove `pin_memory=False` restriction
- Enable AMP: wrap forward pass in `torch.cuda.amp.autocast()`
- Increase `num_workers` to 4–8 for faster data loading

---

## Prediction Format

Both CSVs follow the required format:

```
img_id,predicted_label
AnnualCrop_00001.jpg,0
Forest_00001.jpg,1
...
```

Class index mapping (from `Data/label_map.json`):

| Index | Class |
|-------|-------|
| 0 | AnnualCrop |
| 1 | Forest |
| 2 | HerbaceousVegetation |
| 3 | Highway |
| 4 | Industrial |
| 5 | Pasture |
| 6 | PermanentCrop |
| 7 | Residential |
| 8 | River |
| 9 | SeaLake |
