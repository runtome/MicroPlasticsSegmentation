# UNet Architecture

## Overview

U-Net with a **dual-head** design: a segmentation head for binary mask prediction and a parallel classification head for microplastic type prediction.

Supports **two build modes** controlled by `configs/unet.yaml`:

| Mode | Config | Encoder | Weights |
|------|--------|---------|---------|
| A — Custom | `encoder: null` | DoubleConv/Down blocks | Kaiming / Xavier (from scratch) |
| B — Pretrained | `encoder: resnet34` + `pretrained: true` | ResNet34 via smp | ImageNet pretrained |
| B — SMP no pretrain | `encoder: resnet34` + `pretrained: false` | ResNet34 via smp | Random init |

- **Input:** `(B, 3, 640, 640)`
- **Seg output:** `(B, 1, 640, 640)` — binary mask logit per image
- **Cls output:** `(B, 3)` — multi-label class probabilities (Fiber / Fragment / Film)
- **Requires (Mode B):** `pip install segmentation-models-pytorch`

---

## Mode A — Custom Architecture (encoder: null)

### Architecture Diagram

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

### Channel Sizes (features = [64, 128, 256, 512])

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

### Weight Initialization

| Layer type   | Method                                        |
|--------------|-----------------------------------------------|
| Conv2d       | Kaiming normal (`fan_out`, nonlinearity=relu) |
| BatchNorm2d  | weight=1, bias=0                              |
| Linear       | Xavier normal, bias=0                         |

---

## Mode B — Pretrained Encoder via segmentation-models-pytorch

### Architecture Diagram

```
INPUT: (B, 3, 640, 640)
│
├─ ENCODER: ResNet34 (pretrained ImageNet)
│   layer0 →  (B,  64, 160, 160)   f1
│   layer1 →  (B,  64,  80,  80)   f2
│   layer2 →  (B, 128,  40,  40)   f3
│   layer3 →  (B, 256,  20,  20)   f4
│   layer4 →  (B, 512,  10,  10)   f5  ← BOTTLENECK
│
├─── CLASSIFICATION HEAD (from bottleneck f5)
│    AdaptiveAvgPool2d(1)  →  (B, 512, 1, 1)
│    Flatten               →  (B, 512)
│    Linear(512 → 256)     →  ReLU  →  Dropout(0.3)
│    Linear(256 → 3)       →  Sigmoid
│    OUTPUT: (B, 3)
│
└─── DECODER: smp U-Net decoder (skip connections from encoder)
     up from f5+f4  →  256ch
     up from     +f3  →  128ch
     up from     +f2  →   64ch
     up from     +f1  →   32ch
     segmentation_head Conv3×3 → Conv1×1  →  (B, 1, 640, 640)  ← seg logit
```

### Channel Sizes (ResNet34 encoder)

| Stage        | out_ch | Spatial (input 640×640) |
|--------------|--------|-------------------------|
| encoder f1   | 64     | 160 × 160               |
| encoder f2   | 64     | 80 × 80                 |
| encoder f3   | 128    | 40 × 40                 |
| encoder f4   | 256    | 20 × 20                 |
| encoder f5   | 512    | 10 × 10  (bottleneck)   |

### Parameters Comparison

| Mode | Encoder | Total params |
|------|---------|-------------|
| A — Custom | DoubleConv (scratch) | ~17.4 M |
| B — smp ResNet34 | ResNet34 (ImageNet) | ~24.6 M |

### Weight Initialization (Mode B)

| Component | Init |
|-----------|------|
| ResNet34 encoder | ImageNet pretrained weights (frozen-capable but unfrozen by default) |
| smp decoder | smp default init |
| Classification head | Xavier normal (Linear), bias=0 |

> Only the classification head is re-initialized — pretrained encoder weights are preserved.

### reset_weights() in CV

For 5-fold CV, `reset_weights()` behaves differently per mode:

| Mode | reset_weights() action |
|------|------------------------|
| A — Custom | `_init_weights()` — Kaiming/Xavier reinit all layers |
| B — Pretrained | `_build_smp()` — full re-instantiation, reloads ImageNet weights |

---

## Building Blocks (Mode A)

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

## Classification Head (both modes)

Parallel branch attached to the bottleneck, not the decoder output.

```
bottleneck: (B, 512, H', W')
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

## Config Reference (`configs/unet.yaml`)

```yaml
model:
  name: unet
  encoder: resnet34   # smp encoder name — set to null for custom scratch build
  pretrained: true    # true = ImageNet weights via smp; false = random init
  in_channels: 3
  num_classes: 3
  features: [64, 128, 256, 512]   # used only when encoder: null
  dropout: 0.3
```

### Switching between modes

```yaml
# Mode A — custom from scratch (original behavior)
encoder: null
pretrained: false

# Mode B — pretrained ResNet34 encoder
encoder: resnet34
pretrained: true

# Other supported smp encoders
encoder: resnet50
encoder: efficientnet-b4
encoder: mobilenet_v2
```

---

## Pretrained Encoder Selection Guide

### Recommended encoders for microplastics segmentation

```yaml
# ── Lightweight — fast training, good for experimentation ──────────────────
encoder: resnet18          # 11M params
encoder: mobilenet_v2      # 3.4M params
encoder: efficientnet-b0   # 5.3M params
encoder: mobileone_s0      # lightest option

# ── Medium — good balance (current default) ────────────────────────────────
encoder: resnet34          # 21M params  ← current
encoder: resnet50          # 25M params
encoder: efficientnet-b2   # 9.1M params
encoder: densenet121       # 8M params

# ── Heavy — best accuracy, slower training ─────────────────────────────────
encoder: resnet101         # 44M params
encoder: efficientnet-b4   # 19M params
encoder: se_resnet50       # 28M params  (squeeze-excitation attention)
encoder: resnext50_32x4d   # 25M params  (grouped convolutions)
```

---

### Available pretrained weights

`pretrained: true` always loads `imagenet` weights. Other weight types exist for specific encoders:

| Weight type | Description | Supported encoders |
|---|---|---|
| `imagenet` | Standard ImageNet-1k | **All encoders** |
| `ssl` | Semi-Supervised Learning (Facebook) | `resnet18`, `resnet50`, `resnext50_32x4d`, `resnext101_32x4d/8d` |
| `swsl` | Semi-Weakly Supervised Learning (Facebook) | same as ssl |
| `instagram` | Trained on 1 billion Instagram images | `resnext101_32x8d/16d/32d/48d` |
| `advprop` | Adversarial Propagation training | `efficientnet-b0` to `b8` |
| `noisy-student` | Noisy Student self-training | `timm-efficientnet-b0` to `b8`, `l2` |

> To use weights other than `imagenet`, modify `_build_smp()` in `models/unet/unet.py` to pass the weight string directly: `encoder_weights="noisy-student"` etc.

---

### Full encoder list (76 encoders)

```
# ResNet
resnet18, resnet34, resnet50, resnet101, resnet152

# ResNeXt
resnext50_32x4d, resnext101_32x4d, resnext101_32x8d,
resnext101_32x16d, resnext101_32x32d, resnext101_32x48d

# EfficientNet (native smp)
efficientnet-b0, efficientnet-b1, efficientnet-b2, efficientnet-b3,
efficientnet-b4, efficientnet-b5, efficientnet-b6, efficientnet-b7

# EfficientNet (timm — more weight options)
timm-efficientnet-b0 .. timm-efficientnet-b8, timm-efficientnet-l2
timm-tf_efficientnet_lite0 .. timm-tf_efficientnet_lite4

# DenseNet
densenet121, densenet169, densenet201, densenet161

# VGG (+ batch-norm variants)
vgg11, vgg11_bn, vgg13, vgg13_bn,
vgg16, vgg16_bn, vgg19, vgg19_bn

# Squeeze-and-Excitation (SE)
se_resnet50, se_resnet101, se_resnet152,
se_resnext50_32x4d, se_resnext101_32x4d, senet154

# DPN (Dual Path Networks)
dpn68, dpn68b, dpn92, dpn98, dpn107, dpn131

# Inception
inceptionresnetv2, inceptionv4

# MobileNet / Xception
mobilenet_v2, xception

# MobileOne
mobileone_s0, mobileone_s1, mobileone_s2, mobileone_s3, mobileone_s4

# Mix Transformer — SegFormer encoders
mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5

# timm Selective Kernel
timm-skresnet18, timm-skresnet34, timm-skresnext50_32x4d
```

---

### Example: switch to EfficientNet-B4

```yaml
# configs/unet.yaml
model:
  encoder: efficientnet-b4
  pretrained: true          # loads imagenet weights
  in_channels: 3
  num_classes: 3
  dropout: 0.3
```

No other code changes needed — `UNet._build_smp()` automatically reads the bottleneck channel count from the encoder via `smp_model.encoder.out_channels[-1]`.

---

## Training Configuration

| Setting          | Value                                           |
|------------------|-------------------------------------------------|
| Optimizer        | Adam, lr=1e-3, wd=1e-4                          |
| Batch size       | 4                                               |
| Max epochs       | 50                                              |
| Cross-validation | 5-fold CV                                       |
| LR scheduler     | ReduceLROnPlateau ×0.5 / patience=5 / min=1e-6  |
| Early stopping   | patience=10, monitors val_loss                  |
| Checkpoint       | saves best val_miou → `outputs/checkpoints/unet/` |
| Dropout          | 0.3 (bottleneck + cls head)                     |
| Gradient clip    | max_norm=1.0                                    |

---

## Training Loop Flow

```
fit_kfold()                          # 5-fold CV
└── for fold in 0..4:
    ├── model.reset_weights()        # Mode A: Kaiming/Xavier reinit
    │                                # Mode B: reload ImageNet weights
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
| `models/unet/unet.py` | UNet class — both modes, forward, train_step, val_step, predict |
| `models/unet/blocks.py` | DoubleConv, Down, Up, OutConv, ResidualBlock (Mode A) |
| `models/base_model.py` | BaseModel interface |
| `training/trainer.py` | Trainer — training loop, CV, callbacks |
| `training/losses.py` | CombinedLoss (Dice + BCE) |
| `training/metrics.py` | compute_iou, MetricTracker |
| `training/callbacks.py` | EarlyStopping, ModelCheckpoint |
| `configs/unet.yaml` | Model + training config |
| `configs/base.yaml` | Shared base config (paths, image_size, device) |
