# Mask R-CNN Architecture

## Overview

Mask R-CNN is a **two-stage instance segmentation** model. Unlike the U-Net family (which produces a single binary mask for the whole image), Mask R-CNN detects individual object instances and predicts a separate mask, bounding box, and class label **per instance**.

This implementation wraps `torchvision.models.detection.maskrcnn_resnet50_fpn` with custom head replacements for the microplastics dataset.

- **Input:** list of images `[(3, H, W), ...]` — variable size supported
- **Output (train):** loss dict (box + class + mask losses)
- **Output (eval):** per-image list of `{masks, labels, scores, boxes}` — one entry per detected instance
- **Pretrained backbone:** Yes — **COCO pretrained ResNet-50 + FPN weights** loaded by default
- **num_classes:** 4 (background + Fiber + Fragment + Film)

---

## Architecture Overview

```
INPUT: list of images (B images, each 3×H×W)
│
├─ BACKBONE: ResNet-50 + FPN
│   ResNet-50 stages → C2, C3, C4, C5 feature maps
│   FPN → P2, P3, P4, P5, P6  (multi-scale feature pyramid)
│
├─ REGION PROPOSAL NETWORK (RPN)
│   Slides anchor boxes over each FPN level
│   → Objectness scores + box deltas
│   → ~2000 region proposals (NMS filtered)
│
├─ ROI ALIGN
│   Crops + aligns proposal regions from FPN features
│   → Fixed-size feature maps per proposal
│
├─ BOX HEAD (FastRCNN predictor — REPLACED)
│   FC → FC → [cls_score (num_classes), bbox_pred (num_classes×4)]
│   → Final class + refined box per proposal
│
└─ MASK HEAD (MaskRCNN predictor — REPLACED)
    Conv×4 (256ch each) → ConvTranspose2d (upsample ×2) → Conv1×1(num_classes)
    → (num_classes, 28, 28) binary mask per instance
```

---

## Head Replacements

The original torchvision heads are trained for 91 COCO classes. This wrapper replaces them for `num_classes=4` (background + 3 microplastic classes).

### Box Predictor (FastRCNNPredictor)
```python
# Original: in_features → 91 classes
# Replaced: in_features → 4 classes
self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes=4)
```

### Mask Predictor (MaskRCNNPredictor)
```python
# Original: in_features_mask → 91-class masks
# Replaced: in_features_mask → hidden(256) → 4-class masks
self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
    in_features_mask, hidden_layer=256, num_classes=4
)
```

All other layers (backbone, FPN, RPN, ROI Align, box head FC layers) keep their **COCO pretrained weights**.

---

## Backbone: ResNet-50 + FPN

| Stage   | Output channels | Stride |
|---------|----------------|--------|
| C2 (layer1) | 256       | 4      |
| C3 (layer2) | 512       | 8      |
| C4 (layer3) | 1024      | 16     |
| C5 (layer4) | 2048      | 32     |

FPN lateral + top-down connections → `P2, P3, P4, P5` all at 256 channels, plus `P6` (max-pooled from P5).

> Config says `backbone: resnet101` but the implementation uses **ResNet-50** — torchvision's official pretrained Mask R-CNN only offers R50+FPN weights.

---

## Two-Stage Detection Flow

### Stage 1 — Region Proposal Network (RPN)
- Runs on each FPN level (P2–P6)
- Predicts objectness score + box offset for each anchor
- NMS → top-2000 proposals during training, top-1000 during eval

### Stage 2 — Detection + Segmentation
1. **ROI Align** — crops 7×7 feature maps from FPN for each proposal
2. **Box head** — 2× FC(1024) → class scores + box refinement
3. **NMS** → final instance detections
4. **ROI Align (mask)** — crops 14×14 features for surviving detections
5. **Mask head** — 4× Conv3×3(256) → ConvTranspose2d(256→256, stride=2) → Conv1×1(num_classes)
   → `(num_classes, 28, 28)` soft mask per instance

---

## Loss Function

Torchvision returns the loss dict directly from the model in train mode. No custom loss needed.

| Component       | Formula        | Monitors           |
|-----------------|----------------|--------------------|
| `loss_rpn_box_reg`   | Smooth L1      | RPN box regression |
| `loss_objectness`    | Binary CE      | RPN foreground/bg  |
| `loss_classifier`    | Cross-Entropy  | Instance class     |
| `loss_box_reg`       | Smooth L1      | Box regression     |
| `loss_mask`          | Binary CE      | Per-instance mask  |

```python
total_loss = loss_rpn_box_reg + loss_objectness + loss_classifier + loss_box_reg + loss_mask
```

---

## Target Format

Torchvision requires a specific target dict format per image:

```python
{
    "masks":  (N, H, W)  uint8  — binary instance masks
    "labels": (N,)       int64  — class IDs (1=Fiber, 2=Fragment, 3=Film)
    "boxes":  (N, 4)     float  — [x1, y1, x2, y2] bounding boxes
}
```

The `_prepare_targets()` method in the wrapper converts the standard batch format to this structure.

---

## Val Step Behavior

Mask R-CNN is special: it **needs train mode to compute losses** but **eval mode to produce predictions**.

```python
def val_step():
    model.train()
    loss_dict = model(images, targets)   # get losses

    model.eval()
    preds = model(images)                # get instance predictions
    compute iou(top-1 pred mask vs union of GT masks)

    model.train()                        # reset back to train
```

---

## Predict Output

```python
{
    "masks":  (N, H, W) bool    — binary masks, one per detected instance
    "labels": (N,)      int     — class IDs (1–3)
    "scores": (N,)      float   — confidence scores [0,1]
    "boxes":  (N, 4)    float   — [x1, y1, x2, y2]
}
```

Threshold default = 0.5 applied to soft mask logits.

---

## reset_weights()

Unlike U-Net family (which calls `_init_weights()`), Mask R-CNN reloads **COCO pretrained weights** on each CV fold reset:
```python
def reset_weights(self):
    self.__init__(self.config)   # full re-instantiation with COCO weights
```

---

## Training Configuration (`configs/mask_rcnn.yaml`)

| Setting          | Value                                       |
|------------------|---------------------------------------------|
| Optimizer        | SGD, lr=1e-3, momentum=0.9, wd=5e-4         |
| Batch size       | 2                                           |
| Max epochs       | 50                                          |
| Cross-validation | **No** (use_5fold_cv: false)                |
| LR scheduler     | ReduceLROnPlateau ×0.1 / patience=5         |
| Early stopping   | patience=10, monitors val_loss              |
| Checkpoint       | saves best val_map50 → `outputs/checkpoints/mask_rcnn/` |
| Pretrained       | COCO weights (ResNet-50 FPN backbone + heads) |
| Gradient clip    | max_norm=1.0                                |

---

## Comparison: Mask R-CNN vs U-Net Family

| Feature               | U-Net / AttUNet / DynRUNext | Mask R-CNN              |
|-----------------------|-----------------------------|-------------------------|
| Paradigm              | Semantic segmentation       | Instance segmentation   |
| Output                | 1 binary mask per image     | N masks + boxes + labels per image |
| Backbone              | Custom (from scratch)       | ResNet-50 FPN (COCO pretrained) |
| Handles overlap       | No (union of masks)         | Yes (per-instance)      |
| Loss                  | Dice + BCE                  | box + cls + mask (torchvision) |
| Batch size            | 4                           | 2 (heavier model)       |
| Cross-validation      | 5-fold                      | None                    |
| Optimizer             | Adam                        | SGD + momentum          |

---

## File References

| File | Purpose |
|------|---------|
| `models/mask_rcnn/mask_rcnn_wrapper.py` | MaskRCNNWrapper class |
| `configs/mask_rcnn.yaml` | Model + training config |
| `training/trainer.py` | Universal trainer (train/val loop) |
| `training/metrics.py` | compute_iou for val mIoU |
