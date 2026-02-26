"""
Mask2Former wrapper using HuggingFace Transformers.
Fine-tunes facebook/mask2former-swin-base-coco-instance for instance segmentation.
"""
from typing import Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import (
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
)

from ..base_model import BaseModel
from training.metrics import compute_iou


class Mask2FormerWrapper(BaseModel):
    """
    Mask2Former fine-tuned for 3-class instance segmentation.
    Uses HuggingFace Mask2FormerForUniversalSegmentation.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        model_cfg = config.get("model", config)
        pretrained = model_cfg.get(
            "pretrained_model", "facebook/mask2former-swin-base-coco-instance"
        )
        num_classes = model_cfg.get("num_classes", 3)
        id2label = model_cfg.get("id2label", {0: "Fiber", 1: "Fragment", 2: "Film"})
        label2id = model_cfg.get("label2id", {"Fiber": 0, "Fragment": 1, "Film": 2})

        id2label = {int(k): v for k, v in id2label.items()}

        self.processor = Mask2FormerImageProcessor.from_pretrained(
            pretrained, ignore_index=255, reduce_labels=False
        )

        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            pretrained,
            num_labels=num_classes,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
        self.num_classes = num_classes
        self.image_size = config.get("data", {}).get("image_size", 640)

    def forward(self, pixel_values, mask_labels=None, class_labels=None):
        return self.model(
            pixel_values=pixel_values,
            mask_labels=mask_labels,
            class_labels=class_labels,
        )

    def _prepare_m2f_targets(
        self,
        gt_masks: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        device: torch.device,
    ):
        """Prepare mask_labels and class_labels for Mask2Former."""
        mask_labels_list = []
        class_labels_list = []

        for masks_i, labels_i in zip(gt_masks, gt_labels):
            if masks_i.shape[0] == 0:
                # Empty — use dummy single background mask
                H = W = self.image_size
                mask_labels_list.append(
                    torch.zeros(1, H, W, dtype=torch.float32, device=device)
                )
                class_labels_list.append(
                    torch.zeros(1, dtype=torch.long, device=device)
                )
            else:
                # Convert 1-indexed labels to 0-indexed
                class_labels_0idx = (labels_i - 1).to(device)
                mask_labels_list.append(masks_i.to(device).float())
                class_labels_list.append(class_labels_0idx)

        return mask_labels_list, class_labels_list

    def train_step(self, batch: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
        batch = self._batch_to_device(batch, device)
        images = batch["image"]
        gt_masks = batch["masks"]
        gt_labels = batch["labels"]

        mask_labels, class_labels = self._prepare_m2f_targets(gt_masks, gt_labels, device)

        outputs = self.model(
            pixel_values=images,
            mask_labels=mask_labels,
            class_labels=class_labels,
        )
        loss = outputs.loss

        with torch.no_grad():
            iou = 0.0  # M2F loss-only step; compute IoU in val

        return {"loss": loss, "iou": iou}

    @torch.no_grad()
    def val_step(self, batch: Dict, device: torch.device) -> Dict[str, float]:
        batch = self._batch_to_device(batch, device)
        images = batch["image"]
        gt_masks = batch["masks"]
        gt_labels = batch["labels"]

        mask_labels, class_labels = self._prepare_m2f_targets(gt_masks, gt_labels, device)

        outputs = self.model(
            pixel_values=images,
            mask_labels=mask_labels,
            class_labels=class_labels,
        )
        loss = outputs.loss

        # Compute IoU from predicted masks
        B = images.shape[0]
        H, W = images.shape[2], images.shape[3]
        iou_list = []

        try:
            pred_masks_logits = outputs.masks_queries_logits  # (B, Q, H//4, W//4)
            pred_masks = F.interpolate(pred_masks_logits, size=(H, W), mode="bilinear", align_corners=False)
            pred_masks_binary = (pred_masks.sigmoid() > 0.5)

            for i in range(B):
                if gt_masks[i].shape[0] == 0:
                    continue
                gt_union = gt_masks[i].to(device).max(dim=0).values.bool()
                pred_union = pred_masks_binary[i].max(dim=0).values
                iou = compute_iou(
                    pred_union.cpu().numpy(),
                    gt_union.cpu().numpy(),
                )
                iou_list.append(iou)
        except Exception:
            pass

        miou = float(np.mean(iou_list)) if iou_list else 0.0
        return {"loss": float(loss), "miou": miou}

    def predict(self, image: torch.Tensor, threshold: float = 0.5) -> Dict[str, Any]:
        if image.dim() == 3:
            image = image.unsqueeze(0)
        device = next(self.parameters()).device
        image = image.to(device)

        with torch.no_grad():
            outputs = self.model(pixel_values=image)

        H = W = self.image_size
        pred_masks = F.interpolate(
            outputs.masks_queries_logits, size=(H, W), mode="bilinear", align_corners=False
        )
        masks = (pred_masks.sigmoid() > threshold).squeeze(0).cpu()  # (Q, H, W)
        class_logits = outputs.class_queries_logits.squeeze(0)  # (Q, num_classes+1)
        scores, labels = class_logits[:, :-1].softmax(-1).max(-1)

        return {
            "masks": masks,
            "labels": labels.cpu() + 1,  # back to 1-indexed
            "scores": scores.cpu(),
        }

    def reset_weights(self):
        self.__init__(self.config)
