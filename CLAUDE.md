# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Prepare data splits (run once before training)
python scripts/prepare_data.py                   # full: splits + YOLO format
python scripts/prepare_data.py --no-yolo         # splits only (faster, Kaggle-friendly)

# Train
python scripts/train.py --config configs/unet.yaml
python scripts/train.py --config configs/unet.yaml --fold 0      # single CV fold
python scripts/train.py --config configs/unet.yaml --device cpu  # override device
python scripts/train.py --config configs/unet.yaml --resume outputs/checkpoints/unet/ckpt.pth

# Evaluate
python scripts/evaluate.py --config configs/unet.yaml --checkpoint outputs/checkpoints/unet/best.pth
python scripts/evaluate.py --config configs/unet.yaml --checkpoint best.pth --split val

# Inference
python scripts/predict.py --model unet --checkpoint best.pth --input images/
python scripts/predict.py --model unet --checkpoint best.pth --input img.jpg --threshold 0.5

# Compare all trained models (reads outputs/results/*.json → CSV)
python scripts/compare_models.py --output outputs/results/comparison.csv

# TensorBoard
tensorboard --logdir outputs/logs
```

## Architecture

### Config System
All configs inherit from `configs/base.yaml` via `defaults: [base]`. Model configs override only what differs. The base config defines shared paths (`images/`, `annotation.json`, `data_splits/splits.json`), `image_size: 640`, `num_classes: 3`, and training defaults. Scripts load and deep-merge both files.

### Model Interface
Every model (10 total) inherits `BaseModel` (`models/base_model.py`) and implements:
- `forward(x)` — raw tensor output
- `train_step(batch, device) -> {"loss": tensor, ...}` — full forward + loss in one call
- `val_step(batch, device) -> {"loss": float, "miou": float, ...}` — no grad
- `predict(image, threshold=0.5) -> {"masks": ..., "labels": ..., "scores": ...}` — inference
- `reset_weights()` — reinitialises for k-fold CV

Scripts import models by name via a registry dict in `scripts/train.py` (string → module path + class name).

### U-Net Family (3 models)
Dual-head architecture: segmentation head `(B,1,H,W)` + classification head `(B,num_classes)`. Classification head: GlobalAvgPool → FC(256) → ReLU → FC(num_classes) → Sigmoid. Loss: `CombinedLoss = Dice(seg) + BCE(cls)`. Bottleneck channel calculation: `features[-1]*2 // factor` — `Down(features[-1], bottleneck_ch)` (NOT `bottleneck_ch * factor`). Support 5-fold CV via `reset_weights()` between folds.

### Data Pipeline
1. `data/splits.py` — groups Roboflow augmented variants by base name (strips `_jpg.rf.HASH` suffix) before stratified 70/15/15 split; outputs `data_splits/splits.json` with `train/val/test` lists + 5 `folds`
2. `data/dataset.py` — `MicroPlasticsDataset` reads COCO polygon annotations → binary per-instance masks via `cv2.fillPoly`; returns `{image, masks(N,H,W), labels(N), boxes(N,4)}`
3. `data/dataloader.py` — `build_dataloader(split, ..., fold=None)` picks the right image list; uses custom `collate_fn` for variable-length instance lists
4. `data/transforms.py` — train: HFlip + VFlip + Rotate90 + ColorJitter + GaussNoise + normalize; val/test: resize + normalize only

### Trainer
`training/trainer.py::Trainer` is universal across all model types. It calls `model.train_step` / `model.val_step` and handles the full training loop, ReduceLROnPlateau scheduler, EarlyStopping (patience 10), ModelCheckpoint (saves by `val_miou`), and optional TensorBoard logging (lazy import, graceful fallback).

### Batch Format
Batches contain `image: (B,3,640,640)` tensor plus lists-of-tensors for masks/labels/boxes (because each image can have a different number of instances).

### Losses
- U-Net family: `CombinedLoss` = Dice + BCE (dual-head)
- Mask R-CNN: `DetectionLoss` = box (SmoothL1) + class (CE) + mask (BCE) — returned directly by torchvision
- Transformers / foundation models: BCEDiceLoss or cross-entropy depending on model

### Evaluation outputs
`scripts/evaluate.py` saves `outputs/results/{model}_results.json` with keys: `mIoU`, `IoU_1/2/3`, `mAP50`, `mAP75`, `F1_1/2/3`, `F1_macro`, `inference_ms`, `params`. `compare_models.py` aggregates all JSON files into a single CSV.

## Key Facts

- Classes: Fiber (ID 1, 67%), Fragment (ID 2, 28%), Film (ID 3, 5%) — severe imbalance
- All images resized to 640×640 before model input
- `data/` and `training/` `__init__.py` use lazy imports to avoid eager import of albumentations/tensorboard
- Generated artifacts (`data_splits/`, `outputs/`, `*.pth`, `*.pt`) are gitignored
