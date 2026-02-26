# UNet Architecture

## Overview

Custom U-Net with a **dual-head** design: a segmentation head for binary mask prediction and a parallel classification head for microplastic type prediction.

- **Input:** `(B, 3, 640, 640)`
- **Seg output:** `(B, 1, 640, 640)` — binary mask logit per image
- **Cls output:** `(B, 3)` — multi-label class probabilities (Fiber / Fragment / Film)
- **Pretrained encoder:** No — trained from random initialization (Kaiming / Xavier)

---

## Architecture Diagram

```
INPUT: (B, 3, 640, 640)
│
├─ inc    DoubleConv(3   → 64 )  →  (B,  64, 640, 640)   x1
├─ down1  MaxPool + DoubleConv(64  → 128)  →  (B, 128, 320, 320)   x2
├─ down2  MaxPool + DoubleConv(128 → 256)  →  (B, 256, 160, 160)   x3
├─ down3  MaxPool + DoubleConv(256 → 512)  →  (B, 512,  80,  80)   x4
├─ down4  MaxPool + DoubleConv(512 → 512)  →  (B, 512,  40,  40)   x5  ← BOTTLENECK
│         Dropout2d(0.3)
│
├─── CLASSIFICATION HEAD (from bottleneck x5)
│    AdaptiveAvgPool2d(1)  →  (B, 512, 1, 1)
│    Flatten               →  (B, 512)
│    Linear(512 → 256)     →  ReLU  →  Dropout(0.3)
│    Linear(256 → 3)       →  Sigmoid
│    OUTPUT: (B, 3)
│
└─── DECODER (skip connections from encoder)
     up1   Upsample + cat(x5, x4)  →  DoubleConv(1024 → 256)  →  (B, 256,  80,  80)
     up2   Upsample + cat(   , x3)  →  DoubleConv( 512 → 128)  →  (B, 128, 160, 160)
     up3   Upsample + cat(   , x2)  →  DoubleConv( 256 →  64)  →  (B,  64, 320, 320)
     up4   Upsample + cat(   , x1)  →  DoubleConv( 128 →  64)  →  (B,  64, 640, 640)
     outc  Conv1×1(64 → 1)          →  (B,   1, 640, 640)      ← segmentation logit
```

---

## Building Blocks

### DoubleConv
Two consecutive Conv→BN→ReLU operations.
```
Conv3×3 (no bias) → BatchNorm2d → ReLU
Conv3×3 (no bias) → BatchNorm2d → ReLU
```

### Down
Halves spatial resolution (H×W → H/2 × W/2).
```
MaxPool2d(kernel=2) → DoubleConv(in_ch, out_ch)
```

### Up (bilinear mode)
Doubles spatial resolution back and merges with skip connection.
```
Upsample(scale=2, bilinear) → pad to match skip size → cat(skip, upsampled) → DoubleConv
```
- `in_channels` for DoubleConv = skip_ch + upsampled_ch (concatenation doubles channels)
- `mid_channels` = `in_channels // 2` to keep computation balanced

### OutConv
Final 1×1 convolution — outputs raw logit (no activation).
```
Conv1×1(64 → 1)
```
Sigmoid applied at inference: `mask = sigmoid(logit) > 0.5`

---

## Channel Sizes (features = [64, 128, 256, 512])

| Layer   | Operation                  | in_ch | out_ch | Spatial size   |
|---------|----------------------------|-------|--------|----------------|
| inc     | DoubleConv                 | 3     | 64     | 640 × 640      |
| down1   | MaxPool + DoubleConv       | 64    | 128    | 320 × 320      |
| down2   | MaxPool + DoubleConv       | 128   | 256    | 160 × 160      |
| down3   | MaxPool + DoubleConv       | 256   | 512    | 80 × 80        |
| down4   | MaxPool + DoubleConv       | 512   | 512    | 40 × 40        |
| up1     | Upsample + cat + DConv     | 1024  | 256    | 80 × 80        |
| up2     | Upsample + cat + DConv     | 512   | 128    | 160 × 160      |
| up3     | Upsample + cat + DConv     | 256   | 64     | 320 × 320      |
| up4     | Upsample + cat + DConv     | 128   | 64     | 640 × 640      |
| outc    | Conv1×1                    | 64    | 1      | 640 × 640      |

> **Bottleneck channel formula:** `features[-1] * 2 // factor` where `factor=2` (bilinear) → `512 * 2 // 2 = 512`

---

## Classification Head

Parallel branch attached to the bottleneck (x5), not the decoder output.

```
x5: (B, 512, 40, 40)
  → AdaptiveAvgPool2d(1)   →  (B, 512, 1, 1)
  → Flatten                →  (B, 512)
  → Linear(512, 256)       →  (B, 256)
  → ReLU
  → Dropout(0.3)
  → Linear(256, 3)         →  (B, 3)
  → Sigmoid                →  multi-label probabilities
```

Output is **multi-label** (independent sigmoid per class), not softmax.
Classes: `0=Fiber`, `1=Fragment`, `2=Film` (0-indexed internally; dataset uses 1-indexed IDs).

---

## Loss Function

`CombinedLoss = DiceLoss(seg) + BCELoss(cls)`

| Component | Target | Prediction |
|-----------|--------|------------|
| Dice loss | Union of all instance masks `(B, H, W)` | `sigmoid(seg_logit)` |
| BCE loss  | Multi-hot class vector `(B, 3)`          | cls head output      |

---

## Weight Initialization

| Layer type   | Method                                    |
|--------------|-------------------------------------------|
| Conv2d       | Kaiming normal (`fan_out`, nonlinearity=relu) |
| BatchNorm2d  | weight=1, bias=0                          |
| Linear       | Xavier normal, bias=0                     |

No pretrained weights are loaded. The `encoder: resnet34` and `pretrained: true` keys in `configs/unet.yaml` are currently unused by this implementation.

---

## Training Configuration (`configs/unet.yaml`)

| Setting          | Value                        |
|------------------|------------------------------|
| Optimizer        | Adam, lr=1e-3, wd=1e-4       |
| Batch size       | 4                            |
| Max epochs       | 50                           |
| Cross-validation | 5-fold CV                    |
| LR scheduler     | ReduceLROnPlateau ×0.5 / patience=5 / min=1e-6 |
| Early stopping   | patience=10, monitors val_loss |
| Checkpoint       | saves best val_miou → `outputs/checkpoints/unet/` |
| Dropout          | 0.3 (bottleneck + cls head)  |
| Gradient clip    | max_norm=1.0                 |

---

## Training Loop Flow

```
fit_kfold()                          # 5-fold CV
└── for fold in 0..4:
    ├── model.reset_weights()        # reinit weights each fold
    ├── build_dataloader(train, fold=N)
    ├── build_dataloader(val,   fold=N)
    └── fit(train_loader, val_loader)
        └── for epoch in 1..50:
            ├── train_epoch()
            │   └── for batch:
            │       zero_grad → train_step → loss.backward
            │       → clip_grad_norm → optimizer.step
            ├── val_epoch()
            │   └── for batch: val_step (no_grad)
            ├── ReduceLROnPlateau.step(val_loss)
            ├── ModelCheckpoint (saves if val_miou improved)
            └── EarlyStopping (breaks if val_loss stagnates 10 epochs)
```

---

## File References

| File | Purpose |
|------|---------|
| `models/unet/unet.py` | UNet class — forward, train_step, val_step, predict |
| `models/unet/blocks.py` | DoubleConv, Down, Up, OutConv, ResidualBlock |
| `models/base_model.py` | BaseModel interface |
| `training/trainer.py` | Trainer — training loop, CV, callbacks |
| `training/losses.py` | CombinedLoss (Dice + BCE) |
| `training/metrics.py` | compute_iou, MetricTracker |
| `training/callbacks.py` | EarlyStopping, ModelCheckpoint |
| `configs/unet.yaml` | Model + training config |
| `configs/base.yaml` | Shared base config (paths, image_size, device) |
