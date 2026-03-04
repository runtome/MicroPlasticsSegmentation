# Attention U-Net Architecture

## Overview

Attention U-Net extends the standard U-Net by inserting **Attention Gates** on every skip connection. Instead of passing encoder features directly to the decoder, each skip connection is filtered by a learned spatial attention map — suppressing irrelevant background regions and focusing on microplastic features.

Same dual-head design as U-Net: segmentation head + parallel classification head.

Two build modes controlled by config:

| | Mode A — Custom | Mode B — Pretrained (smp) |
|---|---|---|
| **When** | `encoder: null` or `pretrained: false` | `encoder: resnet34` + `pretrained: true` |
| **Encoder** | DoubleConv + Down blocks (from scratch) | ResNet-34 (ImageNet weights via smp) |
| **Decoder** | AttentionUp blocks | AttentionUp blocks (same) |
| **Parameters** | 20,111,980 | 27,912,157 |
| **Init** | Kaiming/Xavier everywhere | Pretrained encoder + Kaiming/Xavier decoder |

- **Input:** `(B, 3, 640, 640)`
- **Seg output:** `(B, 1, 640, 640)` — binary mask logit
- **Cls output:** `(B, 3)` — multi-label class probabilities (Fiber / Fragment / Film)

---

## Architecture Diagram — Mode B: Pretrained (default config)

Config: `encoder: resnet34`, `pretrained: true`

```
INPUT: (B, 3, 640, 640)
│
├─── ENCODER (ResNet-34, ImageNet pretrained via smp) ─────────────────────────┐
│                                                                               │
│  stage 0  input pass-through          →  (B,   3, 640, 640)                  │
│  stage 1  conv1+bn+relu+maxpool       →  (B,  64, 320, 320)  ── skip1 ──┐   │
│  stage 2  layer1 (3× BasicBlock)      →  (B,  64, 160, 160)  ── skip2 ─┐│   │
│  stage 3  layer2 (4× BasicBlock)      →  (B, 128,  80,  80)  ── skip3 ┐││   │
│  stage 4  layer3 (6× BasicBlock)      →  (B, 256,  40,  40)  ── skip4│││   │
│  stage 5  layer4 (3× BasicBlock)      →  (B, 512,  20,  20)  BOTTLENECK│││   │
│           Dropout2d(0.3)                                          ││││   │
│                                                                   ││││   │
├─── CLASSIFICATION HEAD (from bottleneck) ◄────────────────────────┘│││   │
│    AdaptiveAvgPool2d(1)  →  (B, 512, 1, 1)                        │││   │
│    Flatten               →  (B, 512)                               │││   │
│    Linear(512 → 256)     →  ReLU  →  Dropout(0.3)                 │││   │
│    Linear(256 → 3)       →  Sigmoid                                │││   │
│    OUTPUT: cls_out (B, 3)                                          │││   │
│                                                                     │││   │
└─── DECODER (custom AttentionUp blocks, trained from scratch) ──────┘││   │
                                                                       ││   │
     attn_up0  Upsample(bottleneck)  →  (B, 512,  40,  40)            ││   │
               AttentionGate(g=512, x=skip4=256, F_int=128)  ◄────────┘│   │
               cat(attn_skip4, up) → DoubleConv(768 → 256)             │   │
               →  (B, 256,  40,  40)                                   │   │
                                                                        │   │
     attn_up1  Upsample  →  (B, 256,  80,  80)                         │   │
               AttentionGate(g=256, x=skip3=128, F_int=64)  ◄──────────┘   │
               cat(attn_skip3, up) → DoubleConv(384 → 128)                │
               →  (B, 128,  80,  80)                                      │
                                                                            │
     attn_up2  Upsample  →  (B, 128, 160, 160)                             │
               AttentionGate(g=128, x=skip2=64, F_int=32)  ◄───────────────┘
               cat(attn_skip2, up) → DoubleConv(192 → 64)
               →  (B,  64, 160, 160)

     attn_up3  Upsample  →  (B,  64, 320, 320)
               AttentionGate(g=64, x=skip1=64, F_int=32)  ◄── skip1
               cat(attn_skip1, up) → DoubleConv(128 → 64)
               →  (B,  64, 320, 320)

     outc      Conv1×1(64 → 1)  →  (B, 1, 320, 320)
     Bilinear upsample to input size  →  (B, 1, 640, 640)  ← segmentation logit
```

---

## Architecture Diagram — Mode A: Custom (from scratch)

Config: `encoder: null` or omitted, `features: [64, 128, 256, 512]`

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
│    OUTPUT: cls_out (B, 3)
│
└─── DECODER (attention-gated skip connections)
     up1   Upsample(x5) → AttentionGate(x4) → cat → DoubleConv(1024 → 512)  →  (B, 512,  80,  80)
     up2   Upsample    → AttentionGate(x3) → cat → DoubleConv( 768 → 256)  →  (B, 256, 160, 160)
     up3   Upsample    → AttentionGate(x2) → cat → DoubleConv( 384 → 128)  →  (B, 128, 320, 320)
     up4   Upsample    → AttentionGate(x1) → cat → DoubleConv( 192 →  64)  →  (B,  64, 640, 640)
     outc  Conv1×1(64 → 1)                         →  (B,   1, 640, 640)   ← segmentation logit
```

---

## Key Differences

| | Mode A (Custom) | Mode B (Pretrained smp) |
|---|---|---|
| Encoder | DoubleConv + Down blocks | ResNet-34 (ImageNet) |
| Encoder channels | [64, 128, 256, 512, 512] | [3, 64, 64, 128, 256, 512] |
| Decoder | 4 AttentionUp blocks | 4 AttentionUp blocks |
| Final upsample | Not needed (output = input size) | Bilinear 320 → 640 (ResNet stem stride) |
| Parameters | 20,111,980 | 27,912,157 |
| Weight init | Kaiming/Xavier all layers | Pretrained encoder + Kaiming/Xavier decoder |
| `reset_weights()` | Reinit all | Reload pretrained encoder + reinit decoder |

### vs Standard U-Net

| | U-Net | Attention U-Net |
|---|---|---|
| Skip connections | Direct concatenation | Filtered by attention gate |
| Upsampling | Bilinear (shared `Up` block) | Bilinear via `AttentionUp` |
| Bottleneck channels (custom) | `features[-1]*2 // factor = 512` | `features[-1] = 512` (no factor division) |
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

**Mode A (Custom):**

| Level | F_g (decoder) | F_l (skip) | F_int |
|-------|--------------|------------|-------|
| up1   | 512          | 512        | 256   |
| up2   | 512          | 256        | 128   |
| up3   | 256          | 128        | 64    |
| up4   | 128          | 64         | 32    |

**Mode B (Pretrained resnet34):**

| Level   | F_g (decoder) | F_l (skip) | F_int |
|---------|--------------|------------|-------|
| attn_up0 | 512          | 256        | 128   |
| attn_up1 | 256          | 128        | 64    |
| attn_up2 | 128          | 64         | 32    |
| attn_up3 | 64           | 64         | 32    |

---

## AttentionUp Block

Combines upsampling, attention gate, and DoubleConv in one module. Shared by both modes.

```
x1 = Upsample(x, scale=2, bilinear)
skip_attended = AttentionGate(g=x1, x=skip)
x1_padded = pad(x1) to match skip size
out = DoubleConv(cat[skip_attended, x1_padded])
```

---

## Channel Sizes — Mode B: Pretrained (resnet34)

| Layer     | Operation                          | in_ch | out_ch | Spatial size  |
|-----------|------------------------------------|-------|--------|---------------|
| stage 1   | conv1+bn+relu+maxpool              | 3     | 64     | 320 × 320     |
| stage 2   | layer1 (3× BasicBlock)             | 64    | 64     | 160 × 160     |
| stage 3   | layer2 (4× BasicBlock)             | 64    | 128    | 80 × 80       |
| stage 4   | layer3 (6× BasicBlock)             | 128   | 256    | 40 × 40       |
| stage 5   | layer4 (3× BasicBlock)             | 256   | 512    | 20 × 20       |
| attn_up0  | Upsample + AttnGate + cat + DConv  | 768   | 256    | 40 × 40       |
| attn_up1  | Upsample + AttnGate + cat + DConv  | 384   | 128    | 80 × 80       |
| attn_up2  | Upsample + AttnGate + cat + DConv  | 192   | 64     | 160 × 160     |
| attn_up3  | Upsample + AttnGate + cat + DConv  | 128   | 64     | 320 × 320     |
| outc      | Conv1×1                            | 64    | 1      | 320 × 320     |
| upsample  | Bilinear interpolation             | 1     | 1      | 640 × 640     |

## Channel Sizes — Mode A: Custom (features = [64, 128, 256, 512])

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

> **Bottleneck (custom):** `features[-1] = 512` (no bilinear factor division — differs from plain U-Net)

---

## Classification Head

Identical to U-Net — parallel branch from bottleneck features.

```
bottleneck: (B, 512, H, W)
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

**Mode A (Custom):** All layers initialized.

| Layer type  | Method                                      |
|-------------|---------------------------------------------|
| Conv2d      | Kaiming normal (`fan_out`, relu)            |
| BatchNorm2d | weight=1, bias=0                            |
| Linear      | Xavier normal, bias=0                       |

**Mode B (Pretrained):** Encoder weights preserved, decoder + cls head initialized.

| Component    | Method                                      |
|--------------|---------------------------------------------|
| Encoder      | ImageNet pretrained (frozen init)           |
| Decoder Conv2d | Kaiming normal (`fan_out`, relu)          |
| Decoder BN   | weight=1, bias=0                            |
| Cls head Linear | Xavier normal, bias=0                    |

---

## Training Configuration (`configs/attention_unet.yaml`)

| Setting          | Value                                      |
|------------------|--------------------------------------------|
| Encoder          | resnet34 (ImageNet pretrained via smp)     |
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
| `models/attention_unet/attention_unet.py` | AttentionUNet class (dual-mode: custom + smp) |
| `models/attention_unet/attention_gate.py` | AttentionGate module |
| `models/unet/blocks.py` | Shared DoubleConv, Down, OutConv blocks |
| `configs/attention_unet.yaml` | Model + training config |
| `training/losses.py` | CombinedLoss (Dice + BCE) |
| `training/trainer.py` | Trainer — training loop, CV, callbacks |
