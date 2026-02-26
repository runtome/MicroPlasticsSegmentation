"""
EfficientSAM wrapper.
EfficientSAM is a lightweight SAM variant using masked image pretraining.
Fine-tuning strategy: freeze image encoder, train mask decoder + prompt encoder.

Installation: pip install git+https://github.com/yformer/EfficientSAM.git
Or use the HuggingFace version: ybelkada/segment-anything
"""
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..base_model import BaseModel
from training.losses import DiceLoss, FocalLoss
from training.metrics import compute_iou


class EfficientSAMWrapper(BaseModel):
    """
    EfficientSAM wrapper for microplastics instance segmentation.
    Falls back to a lightweight SAM-like stub if EfficientSAM is not installed.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        model_cfg = config.get("model", config)
        self.variant = model_cfg.get("variant", "efficient_sam_vits")
        self.freeze_image_encoder = model_cfg.get("freeze_image_encoder", True)
        self.image_size = config.get("data", {}).get("image_size", 640)

        self.model = self._load_model()
        self._setup_training()

        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()

    def _load_model(self):
        try:
            import efficient_sam
            if "vits" in self.variant:
                from efficient_sam.build_efficient_sam import build_efficient_sam_vits
                model = build_efficient_sam_vits()
            else:
                from efficient_sam.build_efficient_sam import build_efficient_sam_vitb
                model = build_efficient_sam_vitb()
            self._use_efficient_sam = True
            return model
        except ImportError:
            # Try HuggingFace SAM as fallback
            try:
                from transformers import SamModel
                model = SamModel.from_pretrained("facebook/sam-vit-base")
                self._use_efficient_sam = False
                self._use_hf_sam = True
                return model
            except Exception:
                self._use_efficient_sam = False
                self._use_hf_sam = False
                return self._build_stub()

    def _build_stub(self):
        self._use_efficient_sam = False
        self._use_hf_sam = False
        return nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
        )

    def _setup_training(self):
        if not (self._use_efficient_sam or self._use_hf_sam):
            return
        if self.freeze_image_encoder:
            for name, param in self.model.named_parameters():
                if "image_encoder" in name or "vision_encoder" in name:
                    param.requires_grad = False

    def forward(self, pixel_values, input_points=None, input_boxes=None):
        if self._use_efficient_sam:
            return self.model(
                pixel_values,
                input_points,
                input_labels=torch.ones(pixel_values.shape[0], 1, 1, dtype=torch.int,
                                        device=pixel_values.device)
                if input_points is not None else None,
            )
        elif self._use_hf_sam:
            return self.model(
                pixel_values=pixel_values,
                input_boxes=input_boxes,
            )
        else:
            return self.model(pixel_values), None

    def _get_pred_masks(self, outputs, B, H, W, device):
        """Extract predicted masks from model output."""
        try:
            if self._use_efficient_sam:
                # EfficientSAM returns (masks, iou_predictions)
                masks, _ = outputs
                masks = masks[:, 0]  # best mask
            elif self._use_hf_sam:
                masks = outputs.pred_masks.squeeze(2)
            else:
                masks = outputs[0]

            masks = F.interpolate(masks, size=(H, W), mode="bilinear", align_corners=False)
            return torch.sigmoid(masks.squeeze(1))
        except Exception:
            return torch.zeros(B, H, W, device=device)

    def train_step(self, batch: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
        batch = self._batch_to_device(batch, device)
        images = batch["image"]
        gt_masks = batch["masks"]
        gt_boxes = batch["boxes"]

        B, C, H, W = images.shape
        seg_gt = torch.zeros(B, H, W, device=device)
        for i in range(B):
            if gt_masks[i].shape[0] > 0:
                seg_gt[i] = gt_masks[i].to(device).max(dim=0).values

        input_boxes = None
        if any(gt_boxes[i].shape[0] > 0 for i in range(B)):
            boxes = []
            for i in range(B):
                if gt_boxes[i].shape[0] > 0:
                    boxes.append(gt_boxes[i][:1].to(device))
                else:
                    boxes.append(torch.zeros(1, 4, device=device))
            input_boxes = torch.stack(boxes)

        outputs = self.forward(images, input_boxes=input_boxes)
        pred_sig = self._get_pred_masks(outputs, B, H, W, device)

        loss = self.dice_loss(pred_sig, seg_gt) + 20.0 * self.focal_loss(
            torch.logit(pred_sig.clamp(1e-6, 1 - 1e-6)), seg_gt
        )

        with torch.no_grad():
            iou = compute_iou(
                (pred_sig > 0.5).cpu().numpy().reshape(-1),
                seg_gt.cpu().numpy().reshape(-1),
            )

        return {"loss": loss, "iou": iou}

    @torch.no_grad()
    def val_step(self, batch: Dict, device: torch.device) -> Dict[str, float]:
        batch = self._batch_to_device(batch, device)
        images = batch["image"]
        gt_masks = batch["masks"]
        gt_boxes = batch["boxes"]

        B, C, H, W = images.shape
        seg_gt = torch.zeros(B, H, W, device=device)
        for i in range(B):
            if gt_masks[i].shape[0] > 0:
                seg_gt[i] = gt_masks[i].to(device).max(dim=0).values

        input_boxes = None
        if any(gt_boxes[i].shape[0] > 0 for i in range(B)):
            boxes = []
            for i in range(B):
                if gt_boxes[i].shape[0] > 0:
                    boxes.append(gt_boxes[i][:1].to(device))
                else:
                    boxes.append(torch.zeros(1, 4, device=device))
            input_boxes = torch.stack(boxes)

        outputs = self.forward(images, input_boxes=input_boxes)
        pred_sig = self._get_pred_masks(outputs, B, H, W, device)

        loss = float(self.dice_loss(pred_sig, seg_gt))
        iou = compute_iou(
            (pred_sig > 0.5).cpu().numpy().reshape(-1),
            seg_gt.cpu().numpy().reshape(-1),
        )

        return {"loss": loss, "miou": iou}

    def predict(self, image: torch.Tensor, boxes: torch.Tensor = None, threshold: float = 0.5) -> Dict[str, Any]:
        if image.dim() == 3:
            image = image.unsqueeze(0)
        device = next(self.parameters()).device
        image = image.to(device)

        if boxes is None:
            H = W = self.image_size
            boxes = torch.tensor([[0, 0, W, H]], dtype=torch.float32).unsqueeze(0).to(device)
        else:
            boxes = boxes.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = self.forward(image, input_boxes=boxes)
            B, _, H, W = image.shape
            pred_sig = self._get_pred_masks(outputs, 1, H, W, device)

        masks = (pred_sig > threshold).cpu()
        return {"masks": masks, "scores": torch.ones(masks.shape[0])}

    def reset_weights(self):
        self.__init__(self.config)
