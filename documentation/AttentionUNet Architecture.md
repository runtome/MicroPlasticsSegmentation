# Attention U-Net Architecture

## Overview

Attention U-Net extends the standard U-Net by inserting **Attention Gates** on every skip connection. Instead of passing encoder features directly to the decoder, each skip connection is filtered by a learned spatial attention map — suppressing irrelevant background regions and focusing on microplastic features.

Same dual-head design as U-Net: segmentation head + parallel classification head.

Two build modes controlled by config:

| | Mode A — Custom | Mode B — Pretrained (smp) |
|---|---|---|
| **When** | `encoder: null` or `pretrained: false` | `encoder: efficientnet-b3` + `pretrained: true` |
| **Encoder** | DoubleConv + Down blocks (from scratch) | EfficientNet-B3 (ImageNet weights via smp) |
| **Decoder** | AttentionUp blocks | AttentionUp blocks (same) |
| **Parameters** | 20,111,980 | ~16.5 M |
| **Init** | Kaiming/Xavier everywhere | Pretrained encoder + Kaiming/Xavier decoder |

- **Input:** `(B, 3, 640, 640)`
- **Seg output:** `(B, 1, 640, 640)` — binary mask logit
- **Cls output:** `(B, 2)` — multi-label class probabilities (Fiber / Fragment)

---

## Architecture Diagram — Mode B: Pretrained (default config)

Config: `encoder: efficientnet-b3`, `pretrained: true`

```
INPUT: (B, 3, 640, 640)
│
├─── ENCODER (EfficientNet-B3, ImageNet pretrained via smp) ───────────────────┐
│                                                                               │
│  stage 0  input pass-through          →  (B,   3, 640, 640)                  │
│  stage 1  stem conv + bn              →  (B,  40, 320, 320)  ── skip1 ──┐   │
│  stage 2  MBConv block 1              →  (B,  32, 160, 160)  ── skip2 ─┐│   │
│  stage 3  MBConv block 2              →  (B,  48,  80,  80)  ── skip3 ┐││   │
│  stage 4  MBConv block 3              →  (B, 136,  40,  40)  ── skip4│││   │
│  stage 5  MBConv block 4              →  (B, 384,  20,  20)  BOTTLENECK│││   │
│           Dropout2d(0.3)                                          ││││   │
│                                                                   ││││   │
├─── CLASSIFICATION HEAD (from bottleneck) ◄────────────────────────┘│││   │
│    AdaptiveAvgPool2d(1)  →  (B, 384, 1, 1)                        │││   │
│    Flatten               →  (B, 384)                               │││   │
│    Linear(384 → 256)     →  ReLU  →  Dropout(0.3)                 │││   │
│    Linear(256 → 3)       →  Sigmoid                                │││   │
│    OUTPUT: cls_out (B, 3)                                          │││   │
│                                                                     │││   │
└─── DECODER (custom AttentionUp blocks, trained from scratch) ──────┘││   │
                                                                       ││   │
     attn_up0  Upsample(bottleneck)  →  (B, 384,  40,  40)            ││   │
               AttentionGate(g=384, x=skip4=136, F_int=68)  ◄─────────┘│   │
               cat(attn_skip4, up) → DoubleConv(520 → 136)             │   │
               →  (B, 136,  40,  40)                                   │   │
                                                                        │   │
     attn_up1  Upsample  →  (B, 136,  80,  80)                         │   │
               AttentionGate(g=136, x=skip3=48, F_int=24)  ◄───────────┘   │
               cat(attn_skip3, up) → DoubleConv(184 → 48)                 │
               →  (B,  48,  80,  80)                                      │
                                                                            │
     attn_up2  Upsample  →  (B,  48, 160, 160)                             │
               AttentionGate(g=48, x=skip2=32, F_int=16)  ◄────────────────┘
               cat(attn_skip2, up) → DoubleConv(80 → 32)
               →  (B,  32, 160, 160)

     attn_up3  Upsample  →  (B,  32, 320, 320)
               AttentionGate(g=32, x=skip1=40, F_int=20)  ◄── skip1
               cat(attn_skip1, up) → DoubleConv(72 → 40)
               →  (B,  40, 320, 320)

     outc      Conv1×1(40 → 1)  →  (B, 1, 320, 320)
     Bilinear upsample to input size  →  (B, 1, 640, 640)  ← segmentation logit
```

---

## Architecture Diagram — Mode A: Custom (from scratch)

Config: `encoder: null` or omitted, `features: [64, 128, 256, 512]`

```
INPUT: (B, 3, 640, 640)
│
├─── ENCODER (custom DoubleConv + Down blocks, trained from scratch) ──────────┐
│                                                                               │
│  inc    DoubleConv(3   → 64 )             →  (B,  64, 640, 640)  ── x1 ──┐  │
│  down1  MaxPool + DoubleConv(64  → 128)   →  (B, 128, 320, 320)  ── x2 ─┐│  │
│  down2  MaxPool + DoubleConv(128 → 256)   →  (B, 256, 160, 160)  ── x3 ┐││  │
│  down3  MaxPool + DoubleConv(256 → 512)   →  (B, 512,  80,  80)  ── x4│││  │
│  down4  MaxPool + DoubleConv(512 → 512)   →  (B, 512,  40,  40)  BOTTLENECK│
│         Dropout2d(0.3)                                          x5 ││││  │
│                                                                     ││││  │
├─── CLASSIFICATION HEAD (from bottleneck x5) ◄───────────────────────┘│││  │
│    AdaptiveAvgPool2d(1)  →  (B, 512, 1, 1)                          │││  │
│    Flatten               →  (B, 512)                                  │││  │
│    Linear(512 → 256)     →  ReLU  →  Dropout(0.3)                   │││  │
│    Linear(256 → 3)       →  Sigmoid                                   │││  │
│    OUTPUT: cls_out (B, 3)                                             │││  │
│                                                                        │││  │
└─── DECODER (custom AttentionUp blocks, trained from scratch) ─────────┘││  │
                                                                          ││  │
     up1   Upsample(x5)  →  (B, 512,  80,  80)                          ││  │
           AttentionGate(g=512, x=x4=512, F_int=256)  ◄─────────────────┘│  │
           cat(attn_x4, up) → DoubleConv(1024 → 512)                     │  │
           →  (B, 512,  80,  80)                                          │  │
                                                                           │  │
     up2   Upsample  →  (B, 512, 160, 160)                                │  │
           AttentionGate(g=512, x=x3=256, F_int=128)  ◄───────────────────┘  │
           cat(attn_x3, up) → DoubleConv(768 → 256)                         │
           →  (B, 256, 160, 160)                                             │
                                                                               │
     up3   Upsample  →  (B, 256, 320, 320)                                    │
           AttentionGate(g=256, x=x2=128, F_int=64)  ◄────────────────────────┘
           cat(attn_x2, up) → DoubleConv(384 → 128)
           →  (B, 128, 320, 320)

     up4   Upsample  →  (B, 128, 640, 640)
           AttentionGate(g=128, x=x1=64, F_int=32)  ◄── x1
           cat(attn_x1, up) → DoubleConv(192 → 64)
           →  (B,  64, 640, 640)

     outc  Conv1×1(64 → 1)  →  (B, 1, 640, 640)  ← segmentation logit
```

### Encoder Detail (features = [64, 128, 256, 512])

Each `DoubleConv` block applies two consecutive Conv3×3 → BN → ReLU operations.
Each `Down` block halves spatial resolution via MaxPool2d(kernel=2) before DoubleConv.

```
inc:   Conv3×3(3→64,  no bias) → BN(64)  → ReLU → Conv3×3(64→64,   no bias) → BN(64)  → ReLU
down1: MaxPool(2) → Conv3×3(64→128, no bias) → BN(128) → ReLU → Conv3×3(128→128, no bias) → BN(128) → ReLU
down2: MaxPool(2) → Conv3×3(128→256, no bias) → BN(256) → ReLU → Conv3×3(256→256, no bias) → BN(256) → ReLU
down3: MaxPool(2) → Conv3×3(256→512, no bias) → BN(512) → ReLU → Conv3×3(512→512, no bias) → BN(512) → ReLU
down4: MaxPool(2) → Conv3×3(512→512, no bias) → BN(512) → ReLU → Conv3×3(512→512, no bias) → BN(512) → ReLU
       → Dropout2d(0.3)
```

> **Bottleneck channel:** `features[-1] = 512` (no bilinear factor division — differs from plain U-Net where `bottleneck = features[-1]*2 // factor`)

### Decoder Detail

Each `AttentionUp` block: Bilinear upsample(×2) → AttentionGate on skip → pad to match → concatenate → DoubleConv.
Output channels at each level match the corresponding skip connection channels.

```
up1: Upsample(512, ×2) → 80×80   |  AttnGate(g=512, x4=512) → attended_x4  |  cat(512+512)=1024 → DConv(1024→512)
up2: Upsample(512, ×2) → 160×160 |  AttnGate(g=512, x3=256) → attended_x3  |  cat(256+512)= 768 → DConv( 768→256)
up3: Upsample(256, ×2) → 320×320 |  AttnGate(g=256, x2=128) → attended_x2  |  cat(128+256)= 384 → DConv( 384→128)
up4: Upsample(128, ×2) → 640×640 |  AttnGate(g=128, x1= 64) → attended_x1  |  cat( 64+128)= 192 → DConv( 192→ 64)
outc: Conv1×1(64 → 1)  → segmentation logit
```

### Attention Gate Parameters (Mode A)

| Level | F_g (decoder) | F_l (skip) | F_int | cat_ch | out_ch |
|-------|--------------|------------|-------|--------|--------|
| up1   | 512          | 512        | 256   | 1024   | 512    |
| up2   | 512          | 256        | 128   | 768    | 256    |
| up3   | 256          | 128        | 64    | 384    | 128    |
| up4   | 128          | 64         | 32    | 192    | 64     |

> **F_int formula (Mode A):** `features[i] // 2` — e.g. `512 // 2 = 256` for up1

### Weight Initialization (Mode A)

All layers initialized from scratch — no pretrained weights.

| Layer type  | Method                                     |
|-------------|------------------------------------------- |
| Conv2d      | Kaiming normal (`fan_out`, relu), bias = 0 |
| BatchNorm2d | weight = 1, bias = 0                       |
| Linear      | Xavier normal, bias = 0                    |

### Parameters (Mode A)

| Component           | Params     |
|---------------------|------------|
| Encoder (inc + down1–4) | ~14.7 M |
| Decoder (up1–4 + outc)  | ~4.9 M  |
| Attention Gates          | ~0.3 M  |
| Classification Head      | ~0.1 M  |
| **Total**                | **20,111,980** |

---

## Key Differences

| | Mode A (Custom) | Mode B (Pretrained smp) |
|---|---|---|
| Encoder | DoubleConv + Down blocks | EfficientNet-B3 (ImageNet) |
| Encoder channels | [64, 128, 256, 512, 512] | [3, 40, 32, 48, 136, 384] |
| Decoder | 4 AttentionUp blocks | 4 AttentionUp blocks |
| Final upsample | Not needed (output = input size) | Bilinear 320 → 640 (EfficientNet stem stride) |
| Parameters | 20,111,980 | ~16.5 M |
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

> Mode A parameters are documented in the [Mode A detail section](#attention-gate-parameters-mode-a) above.

**Mode B (Pretrained efficientnet-b3):**

| Level    | F_g (decoder) | F_l (skip) | F_int |
|----------|--------------|------------|-------|
| attn_up0 | 384          | 136        | 68    |
| attn_up1 | 136          | 48         | 24    |
| attn_up2 | 48           | 32         | 16    |
| attn_up3 | 32           | 40         | 20    |

> **F_int formula (Mode B):** `max(skip_ch // 2, 16)` — e.g. `136 // 2 = 68` for attn_up0

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

## Channel Sizes — Mode B: Pretrained (efficientnet-b3)

| Layer     | Operation                          | in_ch | out_ch | Spatial size  |
|-----------|------------------------------------|-------|--------|---------------|
| stage 1   | stem conv + bn                     | 3     | 40     | 320 × 320     |
| stage 2   | MBConv block 1                     | 40    | 32     | 160 × 160     |
| stage 3   | MBConv block 2                     | 32    | 48     | 80 × 80       |
| stage 4   | MBConv block 3                     | 48    | 136    | 40 × 40       |
| stage 5   | MBConv block 4                     | 136   | 384    | 20 × 20       |
| attn_up0  | Upsample + AttnGate + cat + DConv  | 520   | 136    | 40 × 40       |
| attn_up1  | Upsample + AttnGate + cat + DConv  | 184   | 48     | 80 × 80       |
| attn_up2  | Upsample + AttnGate + cat + DConv  | 80    | 32     | 160 × 160     |
| attn_up3  | Upsample + AttnGate + cat + DConv  | 72    | 40     | 320 × 320     |
| outc      | Conv1×1                            | 40    | 1      | 320 × 320     |
| upsample  | Bilinear interpolation             | 1     | 1      | 640 × 640     |

> Mode A channel sizes are documented in the [Mode A detail section](#architecture-diagram--mode-a-custom-from-scratch) above.

---

## Classification Head

Identical to U-Net — parallel branch from bottleneck features.

```
bottleneck: (B, C, H, W)        # C = 512 (Mode A) or 384 (Mode B, EfficientNet-B3)
  → AdaptiveAvgPool2d(1)   →  (B, C)
  → Linear(C → 256) → ReLU → Dropout(0.3)
  → Linear(256 → 3)  → Sigmoid
  OUTPUT: (B, 3)  multi-label
```

---

## Loss Function

`CombinedLoss = DiceLoss(seg) + BCELoss(cls)` — identical to U-Net.

---

## Weight Initialization

> Mode A weight init is documented in the [Mode A detail section](#weight-initialization-mode-a) above.

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
| Encoder          | efficientnet-b3 (ImageNet pretrained via smp) |
| Optimizer        | AdamW, lr=3e-4, wd=1e-2                    |
| Batch size       | 4                                          |
| Max epochs       | 100                                        |
| Cross-validation | 5-fold CV                                  |
| LR scheduler     | CosineAnnealingLR (T_max=100, min_lr=1e-6) |
| Early stopping   | patience=15, monitors val_loss             |
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
