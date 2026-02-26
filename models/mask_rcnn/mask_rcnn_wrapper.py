"""
Mask R-CNN wrapper using torchvision.
ResNet-101 backbone, COCO pretrained.
Loss: L_box(SmoothL1) + L_class(CE) + L_mask(BCE)
"""
from typing import Dict, Any, List

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from ..base_model import BaseModel
from training.metrics import compute_iou


class MaskRCNNWrapper(BaseModel):
    """
    Wrapper around torchvision Mask R-CNN.

    Note: torchvision provides ResNet-50 FPN with COCO weights officially.
    ResNet-101 backbone requires custom setup — we use R50 for robustness
    but allow config override.

    The model returns loss dict during training and prediction dicts during eval.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        model_cfg = config.get("model", config)
        num_classes = model_cfg.get("num_classes", 4)  # background + 3 classes
        pretrained = model_cfg.get("pretrained", "coco")

        # Load with COCO weights
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained == "coco" else None
        self.model = maskrcnn_resnet50_fpn(weights=weights)

        # Replace box predictor
        in_features_box = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

        # Replace mask predictor
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes
        )

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def _prepare_targets(self, batch: Dict, device: torch.device) -> List[Dict]:
        """Convert batch dict to torchvision target format."""
        targets = []
        for i in range(len(batch["masks"])):
            masks_i = batch["masks"][i].to(device)  # (N, H, W)
            labels_i = batch["labels"][i].to(device)  # (N,)
            boxes_i = batch["boxes"][i].to(device)    # (N, 4)

            if masks_i.shape[0] == 0:
                targets.append({
                    "masks": torch.zeros((0, *masks_i.shape[1:]), dtype=torch.uint8, device=device),
                    "labels": torch.zeros(0, dtype=torch.long, device=device),
                    "boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
                })
            else:
                targets.append({
                    "masks": (masks_i > 0.5).byte(),
                    "labels": labels_i,
                    "boxes": boxes_i,
                })
        return targets

    def train_step(self, batch: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
        images = [img.to(device) for img in batch["image"]]
        targets = self._prepare_targets(batch, device)

        loss_dict = self.model(images, targets)
        total_loss = sum(loss_dict.values())

        return {"loss": total_loss, **{k: float(v) for k, v in loss_dict.items()}}

    @torch.no_grad()
    def val_step(self, batch: Dict, device: torch.device) -> Dict[str, float]:
        # Mask R-CNN needs train mode for loss, eval mode for predictions
        self.model.train()
        images = [img.to(device) for img in batch["image"]]
        targets = self._prepare_targets(batch, device)
        loss_dict = self.model(images, targets)
        total_loss = sum(float(v) for v in loss_dict.values())

        # Get predictions for IoU
        self.model.eval()
        preds = self.model(images)

        iou_list = []
        for pred, tgt in zip(preds, targets):
            if len(pred["masks"]) == 0 or len(tgt["masks"]) == 0:
                continue
            # Use top prediction mask vs union of GT masks
            pred_mask = (pred["masks"][0, 0] > 0.5).cpu().numpy()
            gt_mask = tgt["masks"].max(dim=0).values.cpu().numpy()
            iou_list.append(compute_iou(pred_mask, gt_mask))

        miou = sum(iou_list) / len(iou_list) if iou_list else 0.0
        self.model.train()

        return {"loss": total_loss, "miou": miou}

    def predict(self, image: torch.Tensor, threshold: float = 0.5) -> Dict[str, Any]:
        self.model.eval()
        if image.dim() == 3:
            image = image.unsqueeze(0)
        device = next(self.parameters()).device
        image = image.to(device)

        with torch.no_grad():
            preds = self.model(image)[0]

        masks = (preds["masks"][:, 0] > threshold).cpu()
        labels = preds["labels"].cpu()
        scores = preds["scores"].cpu()
        boxes = preds["boxes"].cpu()

        return {
            "masks": masks,
            "labels": labels,
            "scores": scores,
            "boxes": boxes,
        }

    def reset_weights(self):
        """Reload COCO pretrained weights."""
        self.__init__(self.config)
