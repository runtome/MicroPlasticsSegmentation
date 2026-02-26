# Dynamic RUNext Architecture

## Overview

Dynamic Residual U-Net Extended (DynamicRUNext) upgrades the plain U-Net with three key enhancements:

1. **Residual blocks** — every encoder/decoder stage wraps features through a residual connection to improve gradient flow
2. **Dynamic convolution** — instead of a single fixed kernel, K=4 parallel kernels are weighted by input-conditioned attention (soft kernel selection)
3. **Pixel Shuffle upsampling** — sub-pixel convolution for decoder upsampling instead of bilinear interpolation

Same dual-head design: segmentation head + parallel classification head.

- **Input:** `(B, 3, 640, 640)`
- **Seg output:** `(B, 1, 640, 640)` — binary mask logit
- **Cls output:** `(B, 3)` — multi-label class probabilities (Fiber / Fragment / Film)
- **Pretrained encoder:** No — trained from random initialization (Kaiming / Xavier)

---

## Architecture Diagram

Default config: `base_channels=64`, `depth=4` → `features = [64, 128, 256, 512]`

```
INPUT: (B, 3, 640, 640)
│
├─ stem     ResidualBlock(3 → 64)             →  (B,  64, 640, 640)   enc[0]
├─ downs[0] DynamicResDown(64  → 128)         →  (B, 128, 320, 320)   enc[1]
├─ downs[1] DynamicResDown(128 → 256)         →  (B, 256, 160, 160)   enc[2]
├─ downs[2] DynamicResDown(256 → 512)         →  (B, 512,  80,  80)   enc[3]
│
├─ bottleneck:
│    MaxPool2d(2)                             →  (B, 512,  40,  40)
│    DynamicConv(512 → 1024)                  →  (B,1024,  40,  40)
│    ResidualBlock(1024 → 1024)               →  (B,1024,  40,  40)
│    Dropout2d(0.3)                           →  (B,1024,  40,  40)   x_bottleneck
│
├─── CLASSIFICATION HEAD (from bottleneck)
│    AdaptiveAvgPool2d(1)  →  (B, 1024)
│    Linear(1024 → 256) → ReLU → Dropout(0.3)
│    Linear(256 → 3) → Sigmoid
│    OUTPUT: (B, 3)
│
└─── DECODER
     ups[0]  DynamicResUp(1024, skip=512, out=512)  →  (B, 512, 80,  80)
     ups[1]  DynamicResUp( 512, skip=256, out=256)  →  (B, 256,160, 160)
     ups[2]  DynamicResUp( 256, skip=128, out=128)  →  (B, 128,320, 320)
     ups[3]  DynamicResUp( 128, skip= 64, out= 64)  →  (B,  64,640, 640)
     outc    Conv1×1(64 → 1)                        →  (B,   1,640, 640)  ← logit
```

---

## Building Blocks

### ResidualBlock
Double conv with a 1×1 residual shortcut. Used in stem and all encoder/decoder stages.
```
input x
├── DoubleConv(in_ch → out_ch)            main path
└── Conv1×1(in_ch → out_ch) → BN         shortcut
output = ReLU(main + shortcut)
```

### DynamicConv
K=4 parallel convolutions; output is their weighted sum where weights come from input-conditioned attention.
```
weights = Softmax(Linear(GAP(x), K))    shape: (B, K)
out = Σ_k  weights[:,k] * Conv3×3_k(x)
out = ReLU(BN(out))
```
- `K=4` parallel Conv3×3 kernels
- Attention: `AdaptiveAvgPool2d(1) → Flatten → Linear(in_ch, K) → Softmax`
- Each input gets a different kernel blend — the network learns which kernel style suits each feature map

### DynamicResDown
Encoder down-step: halves spatial size, applies dynamic conv + residual.
```
MaxPool2d(2) → DynamicConv(in → out) → ResidualBlock(out → out)
```

### PixelShuffleUp
Sub-pixel convolution upsampling (sharper than bilinear).
```
Conv3×3(in_ch → out_ch * scale²) → PixelShuffle(scale=2) → BN → ReLU
```
Spatial size doubles: `(B, in_ch, H, W) → (B, out_ch, 2H, 2W)`

### DynamicResUp
Decoder up-step: pixel shuffle + skip connection + residual.
```
x_up = PixelShuffleUp(in_ch → in_ch//2)       doubles spatial
x_up = pad(x_up) to match skip size
out  = ResidualBlock(cat[skip, x_up] → out_ch)
```

---

## Channel Sizes (base_channels=64, depth=4)

`features = [64, 128, 256, 512]`, `bottleneck_ch = features[-1] * 2 = 1024`

| Layer       | Operation                          | in_ch | out_ch | Spatial size  |
|-------------|------------------------------------|-------|--------|---------------|
| stem        | ResidualBlock                      | 3     | 64     | 640 × 640     |
| downs[0]    | DynamicResDown                     | 64    | 128    | 320 × 320     |
| downs[1]    | DynamicResDown                     | 128   | 256    | 160 × 160     |
| downs[2]    | DynamicResDown                     | 256   | 512    | 80 × 80       |
| bottleneck  | MaxPool+DynConv+ResBlock+Dropout   | 512   | 1024   | 40 × 40       |
| ups[0]      | DynamicResUp                       | 1024  | 512    | 80 × 80       |
| ups[1]      | DynamicResUp                       | 512   | 256    | 160 × 160     |
| ups[2]      | DynamicResUp                       | 256   | 128    | 320 × 320     |
| ups[3]      | DynamicResUp                       | 128   | 64     | 640 × 640     |
| outc        | Conv1×1                            | 64    | 1      | 640 × 640     |

> **Bottleneck:** `features[-1] * 2 = 1024` (deeper than U-Net's 512)

---

## Classification Head

Parallel branch from bottleneck — same structure as U-Net/AttentionUNet but with larger input (1024 channels).

```
bottleneck: (B, 1024, 40, 40)
  → AdaptiveAvgPool2d(1)   →  (B, 1024)
  → Linear(1024 → 256) → ReLU → Dropout(0.3)
  → Linear(256 → 3)   → Sigmoid
  OUTPUT: (B, 3)  multi-label
```

---

## DynamicConv Attention Detail

For each forward pass:
```
x: (B, in_ch, H, W)
gap = AdaptiveAvgPool2d(1)(x)       → (B, in_ch, 1, 1)
gap = Flatten(gap)                  → (B, in_ch)
weights = Softmax(Linear(gap, K=4)) → (B, 4)

out = w[:,0]*Conv0(x) + w[:,1]*Conv1(x) + w[:,2]*Conv2(x) + w[:,3]*Conv3(x)
out = ReLU(BN(out))
```

This means **different images in the same batch can use different kernel blends** — the model adapts its convolution style to input content (e.g., elongated Fiber vs rounded Fragment vs flat Film).

---

## Loss Function

`CombinedLoss = DiceLoss(seg) + BCELoss(cls)` — identical to U-Net family.

---

## Weight Initialization

| Layer type  | Method                           |
|-------------|----------------------------------|
| Conv2d      | Kaiming normal (`fan_out`, relu) |
| BatchNorm2d | weight=1, bias=0                 |
| Linear      | Xavier normal, bias=0            |

---

## Training Configuration (`configs/dynamic_runext.yaml`)

| Setting          | Value                                           |
|------------------|-------------------------------------------------|
| Optimizer        | Adam, lr=1e-3, wd=1e-4                          |
| Batch size       | 2 (smaller — model is heavier than U-Net)       |
| Max epochs       | 50                                              |
| Cross-validation | 5-fold CV                                       |
| LR scheduler     | ReduceLROnPlateau ×0.5 / patience=5             |
| Early stopping   | patience=10, monitors val_loss                  |
| Checkpoint       | saves best val_miou → `outputs/checkpoints/dynamic_runext/` |
| Dropout          | 0.3                                             |
| Gradient clip    | max_norm=1.0                                    |

---

## Comparison: U-Net Family

| Feature              | U-Net     | Attention U-Net | DynamicRUNext     |
|----------------------|-----------|-----------------|-------------------|
| Encoder blocks       | DoubleConv | DoubleConv      | DynamicConv + Residual |
| Skip connections     | Direct    | Attention-gated | Direct            |
| Upsampling           | Bilinear  | Bilinear        | Pixel Shuffle     |
| Bottleneck channels  | 512       | 512             | 1024              |
| Adaptive kernels     | No        | No              | Yes (K=4 dynamic) |
| Residual connections | No        | No              | Yes               |
| Batch size           | 4         | 4               | 2                 |

---

## File References

| File | Purpose |
|------|---------|
| `models/dynamic_runext/dynamic_runext.py` | DynamicRUNext class + all custom blocks |
| `models/unet/blocks.py` | Shared ResidualBlock, OutConv |
| `configs/dynamic_runext.yaml` | Model + training config |
| `training/losses.py` | CombinedLoss (Dice + BCE) |
| `training/trainer.py` | Trainer — training loop, CV, callbacks |
