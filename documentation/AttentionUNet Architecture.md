# Attention U-Net Architecture

## Overview

Attention U-Net extends the standard U-Net by inserting **Attention Gates** on every skip connection. Instead of passing encoder features directly to the decoder, each skip connection is filtered by a learned spatial attention map — suppressing irrelevant background regions and focusing on microplastic features.

Same dual-head design as U-Net: segmentation head + parallel classification head.

- **Input:** `(B, 3, 640, 640)`
- **Seg output:** `(B, 1, 640, 640)` — binary mask logit
- **Cls output:** `(B, 3)` — multi-label class probabilities (Fiber / Fragment / Film)
- **Pretrained encoder:** No — trained from random initialization (Kaiming / Xavier)

> Note: `encoder: resnet34` and `pretrained: true` appear in `configs/attention_unet.yaml` but are unused by the implementation.

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
└─── DECODER (attention-gated skip connections)
     up1   Upsample(x5) → AttentionGate(x4) → cat → DoubleConv(1024 → 512)  →  (B, 512,  80,  80)
     up2   Upsample    → AttentionGate(x3) → cat → DoubleConv( 768 → 256)  →  (B, 256, 160, 160)
     up3   Upsample    → AttentionGate(x2) → cat → DoubleConv( 384 → 128)  →  (B, 128, 320, 320)
     up4   Upsample    → AttentionGate(x1) → cat → DoubleConv( 192 →  64)  →  (B,  64, 640, 640)
     outc  Conv1×1(64 → 1)                         →  (B,   1, 640, 640)   ← segmentation logit
```

---

## Key Difference vs Standard U-Net

| | U-Net | Attention U-Net |
|---|---|---|
| Skip connections | Direct concatenation | Filtered by attention gate |
| Upsampling | Bilinear (shared `Up` block) | Bilinear via `AttentionUp` |
| Bottleneck channels | `features[-1]*2 // factor = 512` | `features[-1] = 512` (no factor division) |
| Extra parameters | None | Attention gate weights per decoder level |

---

## Attention Gate

Defined in `models/attention_unet/attention_gate.py`. Produces a spatial attention map `ψ ∈ [0,1]` that scales the skip connection feature map.

```
Inputs:
  g  — gating signal from decoder    (B, F_g, H', W')  ← coarser, semantic
  x  — skip from encoder             (B, F_l,  H,  W)  ← finer, spatial

Steps:
  g_up = Upsample(g) to match x size
  g1   = Conv1×1(g_up) → BN             shape: (B, F_int, H, W)
  x1   = Conv1×1(x)    → BN             shape: (B, F_int, H, W)
  ψ    = Conv1×1(ReLU(g1 + x1)) → BN → Sigmoid   shape: (B, 1, H, W)
  out  = x * ψ                          shape: (B, F_l, H, W)
```

The attention map `ψ` highlights spatial regions relevant to the gating signal — regions with high response are preserved, low-response regions are suppressed.

### Attention Gate Parameters per Decoder Level

| Level | F_g (decoder) | F_l (skip) | F_int |
|-------|--------------|------------|-------|
| up1   | 512          | 512        | 256   |
| up2   | 512          | 256        | 128   |
| up3   | 256          | 128        | 64    |
| up4   | 128          | 64         | 32    |

---

## AttentionUp Block

Combines upsampling, attention gate, and DoubleConv in one module.

```
x1 = Upsample(x, scale=2, bilinear)
skip_attended = AttentionGate(g=x1, x=skip)
x1_padded = pad(x1) to match skip size
out = DoubleConv(cat[skip_attended, x1_padded])
```

---

## Channel Sizes (features = [64, 128, 256, 512])

| Layer  | Operation                          | in_ch | out_ch | Spatial size  |
|--------|------------------------------------|-------|--------|---------------|
| inc    | DoubleConv                         | 3     | 64     | 640 × 640     |
| down1  | MaxPool + DoubleConv               | 64    | 128    | 320 × 320     |
| down2  | MaxPool + DoubleConv               | 128   | 256    | 160 × 160     |
| down3  | MaxPool + DoubleConv               | 256   | 512    | 80 × 80       |
| down4  | MaxPool + DoubleConv               | 512   | 512    | 40 × 40       |
| up1    | Upsample + AttnGate + cat + DConv  | 1024  | 512    | 80 × 80       |
| up2    | Upsample + AttnGate + cat + DConv  | 768   | 256    | 160 × 160     |
| up3    | Upsample + AttnGate + cat + DConv  | 384   | 128    | 320 × 320     |
| up4    | Upsample + AttnGate + cat + DConv  | 192   | 64     | 640 × 640     |
| outc   | Conv1×1                            | 64    | 1      | 640 × 640     |

> **Bottleneck:** `features[-1] = 512` (no bilinear factor division — differs from plain U-Net)

---

## Classification Head

Identical to U-Net — parallel branch from bottleneck x5.

```
x5: (B, 512, 40, 40)
  → AdaptiveAvgPool2d(1)   →  (B, 512)
  → Linear(512 → 256) → ReLU → Dropout(0.3)
  → Linear(256 → 3)  → Sigmoid
  OUTPUT: (B, 3)  multi-label
```

---

## Loss Function

`CombinedLoss = DiceLoss(seg) + BCELoss(cls)` — identical to U-Net.

---

## Weight Initialization

| Layer type  | Method                                      |
|-------------|---------------------------------------------|
| Conv2d      | Kaiming normal (`fan_out`, relu)            |
| BatchNorm2d | weight=1, bias=0                            |
| Linear      | Xavier normal, bias=0                       |

---

## Training Configuration (`configs/attention_unet.yaml`)

| Setting          | Value                                      |
|------------------|--------------------------------------------|
| Optimizer        | Adam, lr=1e-3, wd=1e-4                     |
| Batch size       | 4                                          |
| Max epochs       | 50                                         |
| Cross-validation | 5-fold CV                                  |
| LR scheduler     | ReduceLROnPlateau ×0.5 / patience=5        |
| Early stopping   | patience=10, monitors val_loss             |
| Checkpoint       | saves best val_miou → `outputs/checkpoints/attention_unet/` |
| Dropout          | 0.3                                        |
| Gradient clip    | max_norm=1.0                               |

---

## File References

| File | Purpose |
|------|---------|
| `models/attention_unet/attention_unet.py` | AttentionUNet class |
| `models/attention_unet/attention_gate.py` | AttentionGate module |
| `models/unet/blocks.py` | Shared DoubleConv, Down, OutConv blocks |
| `configs/attention_unet.yaml` | Model + training config |
| `training/losses.py` | CombinedLoss (Dice + BCE) |
| `training/trainer.py` | Trainer — training loop, CV, callbacks |
