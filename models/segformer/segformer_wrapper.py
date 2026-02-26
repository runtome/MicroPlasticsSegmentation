"""
SegFormer wrapper using HuggingFace Transformers.
Fine-tunes nvidia/mit-b2 for semantic segmentation on microplastics (3 classes).
"""
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerConfig

from ..base_model import BaseModel
from training.metrics import compute_iou


class SegFormerWrapper(BaseModel):
    """
    SegFormer fine-tuned for 3-class semantic segmentation.
    Each class gets a separate segmentation channel (0=Fiber, 1=Fragment, 2=Film).
    """

    def __init__(self, config: dict):
        super().__init__(config)
        model_cfg = config.get("model", config)
        pretrained = model_cfg.get("pretrained_model", "nvidia/mit-b2")
        num_classes = model_cfg.get("num_classes", 3)
        id2label = model_cfg.get("id2label", {0: "Fiber", 1: "Fragment", 2: "Film"})
        label2id = model_cfg.get("label2id", {"Fiber": 0, "Fragment": 1, "Film": 2})

        # Convert int keys to int (YAML loads them as str sometimes)
        id2label = {int(k): v for k, v in id2label.items()}

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained,
            num_labels=num_classes,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
        self.num_classes = num_classes
        self.image_size = config.get("data", {}).get("image_size", 640)

    def forward(self, pixel_values: torch.Tensor):
        return self.model(pixel_values=pixel_values)

    def _build_semantic_gt(
        self,
        gt_masks: list,
        gt_labels: list,
        B: int,
        H: int,
        W: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build (B, H, W) semantic segmentation map (0-indexed class, 0=background)."""
        seg_gt = torch.zeros(B, H, W, dtype=torch.long, device=device)
        for i in range(B):
            for mask, lbl in zip(gt_masks[i], gt_labels[i]):
                class_idx = lbl.item()  # 1-indexed → use as-is, 0=background
                seg_gt[i][mask.to(device).bool()] = class_idx
        return seg_gt

    def train_step(self, batch: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
        batch = self._batch_to_device(batch, device)
        images = batch["image"]
        gt_masks = batch["masks"]
        gt_labels = batch["labels"]
        B, C, H, W = images.shape

        seg_gt = self._build_semantic_gt(gt_masks, gt_labels, B, H, W, device)

        outputs = self.model(pixel_values=images, labels=seg_gt)
        loss = outputs.loss

        with torch.no_grad():
            logits = outputs.logits  # (B, num_classes+1, H//4, W//4)
            upsampled = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
            pred = upsampled.argmax(dim=1)  # (B, H, W)
            iou = compute_iou(
                (pred > 0).cpu().numpy().reshape(-1),
                (seg_gt > 0).cpu().numpy().reshape(-1),
            )

        return {"loss": loss, "iou": iou}

    @torch.no_grad()
    def val_step(self, batch: Dict, device: torch.device) -> Dict[str, float]:
        batch = self._batch_to_device(batch, device)
        images = batch["image"]
        gt_masks = batch["masks"]
        gt_labels = batch["labels"]
        B, C, H, W = images.shape

        seg_gt = self._build_semantic_gt(gt_masks, gt_labels, B, H, W, device)
        outputs = self.model(pixel_values=images, labels=seg_gt)
        loss = outputs.loss

        logits = outputs.logits
        upsampled = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        pred = upsampled.argmax(dim=1)

        iou = compute_iou(
            (pred > 0).cpu().numpy().reshape(-1),
            (seg_gt > 0).cpu().numpy().reshape(-1),
        )

        return {"loss": float(loss), "miou": iou}

    def predict(self, image: torch.Tensor, threshold: float = 0.5) -> Dict[str, Any]:
        if image.dim() == 3:
            image = image.unsqueeze(0)
        device = next(self.parameters()).device
        image = image.to(device)

        with torch.no_grad():
            outputs = self.model(pixel_values=image)

        H = W = self.image_size
        logits = outputs.logits
        upsampled = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        pred_class = upsampled.argmax(dim=1).squeeze(0)  # (H, W)

        return {
            "semantic_mask": pred_class.cpu(),
            "logits": upsampled.squeeze(0).cpu(),
        }

    def reset_weights(self):
        model_cfg = self.config.get("model", self.config)
        pretrained = model_cfg.get("pretrained_model", "nvidia/mit-b2")
        self.__init__(self.config)
