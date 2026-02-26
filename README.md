# Microplastics Instance Segmentation

A deep-learning benchmark pipeline for **instance segmentation of microplastic particles** (fibers, fragments, and films) in microscopy images. The codebase trains and evaluates 10 models — ranging from classic U-Nets to modern transformers and YOLO — under identical conditions to produce a fair, reproducible comparison.

---

## Overview

Microplastics are pervasive environmental contaminants. Automated instance segmentation can accelerate large-scale monitoring by replacing manual particle counting. This project:

- Implements **10 segmentation models** (U-Net family, Mask R-CNN, SegFormer, Mask2Former, SAM 2, EfficientSAM, RT-DETR, YOLO26)
- Trains on a real annotated microscopy dataset (607 images, COCO format)
- Evaluates with mIoU, mAP50, mAP75, F1, parameter count, and inference speed
- Supports 5-fold cross-validation and stratified train/val/test splits

---

## Dataset

| Property | Value |
|----------|-------|
| Total annotated images | 607 |
| Unannotated (inference only) | 43 |
| Annotation format | COCO (polygon, single-polygon per instance) |
| Total annotations | 940 |
| Image sizes | 480×640 (majority), some 640×640 and larger — all resized to 640×640 |

**Classes and distribution:**

| ID | Class | Count | Share |
|----|-------|-------|-------|
| 1 | Fiber | 630 | 67.0 % |
| 2 | Fragment | 260 | 27.7 % |
| 3 | Film | 50 | 5.3 % |

Severe class imbalance (Film is rare) makes this a challenging benchmark.

**Splits** (grouped by base image to prevent Roboflow-augment leakage):

| Split | Images |
|-------|--------|
| Train | 421 (70 %) |
| Val | 97 (15 %) |
| Test | 89 (15 %) |

5-fold cross-validation is available for the U-Net family models (paper protocol).

> **Note:** The dataset (`annotation.json` + `images/`) is not included in this repo. See [Quick Start](#quick-start) for setup instructions.

---

## Models

| # | Model | Type | Backbone | Loss | Notes |
|---|-------|------|----------|------|-------|
| 1 | U-Net | Encoder-decoder | VGG-style | Dice + BCE (dual-head) | Dual seg+classify head |
| 2 | Attention U-Net | Encoder-decoder | U-Net + attention gates | Dice + BCE | Soft attention on skip connections |
| 3 | Dynamic R-UNet | Encoder-decoder | Residual + dynamic conv | Dice + BCE | Pixel-shuffle upsample |
| 4 | Mask R-CNN | Two-stage detector | ResNet-101 + FPN | Box + Class + Mask | COCO pretrained |
| 5 | SegFormer | Transformer | MiT-B2 | Cross-entropy | HuggingFace fine-tune |
| 6 | Mask2Former | Transformer | Swin-Base | Bipartite matching | HuggingFace fine-tune |
| 7 | SAM 2 | Foundation model | Hiera | BCE + Dice | Freeze encoder, fine-tune decoder |
| 8 | EfficientSAM | Foundation model | TinyViT | BCE + Dice | Lightweight SAM variant |
| 9 | RT-DETR | Transformer detector | ResNet-50 | Set prediction | HuggingFace, seg head added |
| 10 | YOLO26 | One-stage detector | CSP-DarkNet | YOLO seg loss | Ultralytics (s/m/x variants) |

---

## Project Structure

```
MicroPlasticsSegmentation/
├── configs/
│   ├── base.yaml               # Shared: paths, data settings, metrics
│   ├── unet.yaml               # U-Net hyperparams
│   ├── attention_unet.yaml
│   ├── dynamic_runext.yaml
│   ├── mask_rcnn.yaml
│   ├── segformer.yaml
│   ├── mask2former.yaml
│   ├── sam2.yaml
│   ├── efficient_sam.yaml
│   ├── rtdetr.yaml
│   └── yolo26.yaml
├── data/
│   ├── dataset.py              # MicroPlasticsDataset — COCO polygons → binary masks
│   ├── transforms.py           # Albumentations pipelines
│   ├── splits.py               # Roboflow-aware train/val/test splits + 5-fold CV
│   ├── dataloader.py           # DataLoader factory
│   └── converters/
│       ├── to_mask.py          # COCO polygon → binary mask (cv2.fillPoly)
│       └── to_yolo.py          # COCO → YOLO segmentation format
├── models/
│   ├── base_model.py           # Abstract BaseModel (train_step, val_step, predict)
│   ├── unet/
│   ├── attention_unet/
│   ├── dynamic_runext/
│   ├── mask_rcnn/
│   ├── segformer/
│   ├── mask2former/
│   ├── sam2/
│   ├── efficient_sam/
│   ├── rtdetr/
│   └── yolo26/
├── training/
│   ├── trainer.py              # Universal Trainer (train/val loop, 5-fold CV, TensorBoard)
│   ├── losses.py               # CombinedLoss, FocalLoss, DetectionLoss
│   ├── metrics.py              # IoU, mAP, F1 per class
│   └── callbacks.py            # EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
├── evaluation/
│   ├── evaluator.py            # Load checkpoint → test set → metrics dict
│   └── comparison.py           # Aggregate all models → comparison.csv
├── inference/
│   ├── predictor.py            # Single-image and batch inference
│   └── visualize.py            # Draw masks + class labels
├── scripts/
│   ├── prepare_data.py         # Validate data, create splits.json, convert for YOLO
│   ├── train.py                # Entry: train a single model
│   ├── evaluate.py             # Entry: evaluate a saved checkpoint
│   ├── compare_models.py       # Generate final comparison CSV + table
│   └── predict.py              # Entry: inference on new images
├── requirements.txt
└── setup.py
```

---

## Installation

```bash
git clone https://github.com/USERNAME/MicroPlasticsSegmentation.git
cd MicroPlasticsSegmentation

pip install -r requirements.txt
pip install -e .
```

**Key dependencies:** `torch`, `torchvision`, `transformers`, `albumentations`, `ultralytics`, `opencv-python`, `pycocotools`, `tensorboard`, `pandas`, `scipy`

> Tested on Python 3.10+, PyTorch 2.x, CUDA 11.8+

---

## Quick Start

### Step 1 — Place your dataset

```
MicroPlasticsSegmentation/
├── annotation.json       ← COCO format annotation file
└── images/               ← all 607+ images (jpg/png)
```

### Step 2 — Prepare splits

```bash
python scripts/prepare_data.py
```

Outputs `data_splits/splits.json` (train/val/test image IDs + 5-fold folds) and optionally `data_splits/yolo/` for YOLO training.

### Step 3 — Train a model

```bash
python scripts/train.py --config configs/unet.yaml
```

Checkpoints saved to `outputs/checkpoints/`. TensorBoard logs to `outputs/logs/`.

```bash
# Monitor training
tensorboard --logdir outputs/logs
```

### Step 4 — Evaluate a checkpoint

```bash
python scripts/evaluate.py \
    --config configs/unet.yaml \
    --checkpoint outputs/checkpoints/unet_best.pth
```

### Step 5 — Compare all trained models

```bash
python scripts/compare_models.py --output outputs/results/comparison.csv
```

### Step 6 — Run inference on new images

```bash
python scripts/predict.py \
    --model unet \
    --checkpoint outputs/checkpoints/unet_best.pth \
    --input path/to/images/
```

---

## Config System

Every model has its own YAML that inherits shared settings from `configs/base.yaml`:

```yaml
# configs/base.yaml — shared across all models
data:
  images_dir: images/
  annotation: annotation.json
  splits_file: data_splits/splits.json
  image_size: 640
  num_classes: 3
  class_names: [Fiber, Fragment, Film]
training:
  output_dir: outputs/
  seed: 42
  use_5fold_cv: false
```

Model configs override only what differs:

```yaml
# configs/unet.yaml
model:
  name: unet
  features: [64, 128, 256, 512]
training:
  optimizer: adam
  lr: 1.0e-3
  batch_size: 8
  num_epochs: 1
  use_5fold_cv: true
```

Pass any config to the scripts via `--config configs/<model>.yaml`.

---

## Kaggle Usage

```python
# ── Cell 1: clone & install ────────────────────────────────────────────────
!git clone https://github.com/USERNAME/MicroPlasticsSegmentation.git
%cd MicroPlasticsSegmentation
!pip install -r requirements.txt -q
!pip install -e . -q

# ── Cell 2: copy dataset from Kaggle input ─────────────────────────────────
import shutil, os

os.makedirs("images", exist_ok=True)
!cp /kaggle/input/DATASET_NAME/annotation.json .
!cp /kaggle/input/DATASET_NAME/images/* images/

# ── Cell 3: prepare splits ─────────────────────────────────────────────────
!python scripts/prepare_data.py --no-yolo

# ── Cell 4: 1-epoch smoke test (edit config first) ─────────────────────────
# Option A — edit configs/unet.yaml and set num_epochs: 1
!python scripts/train.py --config configs/unet.yaml --device cuda

# Option B — override from command line (if supported by your argparse setup)
# !python scripts/train.py --config configs/unet.yaml --epochs 1 --device cuda
```

> **GPU note:** Select *GPU T4 x2* or *GPU P100* in *Session options → Accelerator* before running. Training a full U-Net for 50 epochs takes ~2–4 hours on a T4.

> **YOLO weights:** `yolo26s-seg.pt` etc. are auto-downloaded by Ultralytics on first use and cached in `~/.cache/ultralytics/`. They are excluded from this repo via `.gitignore`.

---

## Results

> Placeholder — fill in after training all models.

| Model | mIoU | mAP50 | mAP75 | F1 | Params (M) | Inference (ms/img) |
|-------|------|-------|-------|----|------------|-------------------|
| U-Net | - | - | - | - | - | - |
| Attention U-Net | - | - | - | - | - | - |
| Dynamic R-UNet | - | - | - | - | - | - |
| Mask R-CNN | - | - | - | - | - | - |
| SegFormer | - | - | - | - | - | - |
| Mask2Former | - | - | - | - | - | - |
| SAM 2 | - | - | - | - | - | - |
| EfficientSAM | - | - | - | - | - | - |
| RT-DETR | - | - | - | - | - | - |
| YOLO26-s | - | - | - | - | - | - |
| YOLO26-m | - | - | - | - | - | - |
| YOLO26-x | - | - | - | - | - | - |

Per-class metrics (Fiber / Fragment / Film) are logged to `outputs/results/comparison.csv` after running `compare_models.py`.

---

## Citation / License

If you use this codebase or dataset in your work, please cite:

```
@misc{microplastics-seg-2025,
  title  = {Microplastics Instance Segmentation Benchmark},
  year   = {2025},
  url    = {https://github.com/USERNAME/MicroPlasticsSegmentation}
}
```

This project is released under the **MIT License**.
