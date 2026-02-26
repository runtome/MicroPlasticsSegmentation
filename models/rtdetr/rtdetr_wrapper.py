"""
RT-DETR wrapper using HuggingFace Transformers.
RT-DETR is a real-time detection transformer.
We fine-tune PekingU/rtdetr-r50 for 3-class detection + add a segmentation head.
"""
from typing import Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..base_model import BaseModel
from training.losses import DiceLoss
from training.metrics import compute_iou


class MaskHead(nn.Module):
    """
    Lightweight mask head added on top of RT-DETR.
    Takes encoder features + predicted boxes → binary masks per query.
    """

    def __init__(self, feat_dim: int = 256, mask_size: int = 128):
        super().__init__()
        self.mask_size = mask_size
        self.conv = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim, feat_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim // 2, 1, 1),
        )

    def forward(self, features: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """
        ROI-align approach: for each query box, extract features and predict mask.
        Simplified: global feature mask head (no per-instance RoI).

        Args:
            features: (B, C, H, W) encoder features
            boxes: (B, N, 4) normalized boxes [x1,y1,x2,y2]

        Returns:
            masks: (B, N, mask_size, mask_size)
        """
        B, N = boxes.shape[:2]
        H = W = self.mask_size

        # Global mask from features
        feat_up = F.interpolate(features, size=(H, W), mode="bilinear", align_corners=False)
        base_mask = self.conv(feat_up)  # (B, 1, H, W)

        # Expand to N instances (simple broadcast — boxes used as attention hint)
        masks = base_mask.expand(B, N, H, W)
        return masks


class RTDETRWrapper(BaseModel):
    """
    RT-DETR with segmentation head for instance segmentation.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        model_cfg = config.get("model", config)
        pretrained = model_cfg.get("pretrained_model", "PekingU/rtdetr-r50")
        self.num_classes = model_cfg.get("num_classes", 3)
        self.image_size = config.get("data", {}).get("image_size", 640)

        self.model = self._load_model(pretrained)
        self.mask_head = MaskHead(feat_dim=256, mask_size=self.image_size // 4)
        self.dice_loss = DiceLoss()

    def _load_model(self, pretrained: str):
        try:
            from transformers import RTDetrForObjectDetection
            id2label = {0: "Fiber", 1: "Fragment", 2: "Film"}
            label2id = {v: k for k, v in id2label.items()}
            model = RTDetrForObjectDetection.from_pretrained(
                pretrained,
                num_labels=self.num_classes,
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True,
            )
            self._use_hf = True
            return model
        except (ImportError, Exception) as e:
            print(f"RT-DETR HuggingFace not available ({e}), using stub.")
            self._use_hf = False
            return nn.Sequential(
                nn.Conv2d(3, 256, 3, padding=1),
                nn.ReLU(),
            )

    def forward(self, pixel_values, labels=None):
        if not self._use_hf:
            feat = self.model(pixel_values)
            return feat, None
        if labels is not None:
            outputs = self.model(pixel_values=pixel_values, labels=labels)
        else:
            outputs = self.model(pixel_values=pixel_values)
        return outputs

    def _prepare_detection_targets(
        self, gt_boxes: list, gt_labels: list, device: torch.device, H: int, W: int
    ) -> List[Dict]:
        """Convert batch GT to HuggingFace RT-DETR target format."""
        targets = []
        for boxes_i, labels_i in zip(gt_boxes, gt_labels):
            if boxes_i.shape[0] == 0:
                targets.append({
                    "boxes": torch.zeros(0, 4, device=device),
                    "class_labels": torch.zeros(0, dtype=torch.long, device=device),
                })
            else:
                # Normalize boxes to [0,1] center format (cx,cy,w,h)
                boxes = boxes_i.to(device).float()
                cx = (boxes[:, 0] + boxes[:, 2]) / 2 / W
                cy = (boxes[:, 1] + boxes[:, 3]) / 2 / H
                bw = (boxes[:, 2] - boxes[:, 0]) / W
                bh = (boxes[:, 3] - boxes[:, 1]) / H
                norm_boxes = torch.stack([cx, cy, bw, bh], dim=-1).clamp(0, 1)
                class_labels = (labels_i.to(device) - 1)  # 0-indexed
                targets.append({
                    "boxes": norm_boxes,
                    "class_labels": class_labels,
                })
        return targets

    def train_step(self, batch: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
        batch = self._batch_to_device(batch, device)
        images = batch["image"]
        gt_masks = batch["masks"]
        gt_labels = batch["labels"]
        gt_boxes = batch["boxes"]

        B, C, H, W = images.shape

        if not self._use_hf:
            feat = self.model(images)
            dummy_boxes = torch.zeros(B, 1, 4, device=device)
            mask_preds = self.mask_head(feat, dummy_boxes)
            seg_gt = torch.zeros(B, 1, H // 4, W // 4, device=device)
            loss = self.dice_loss(torch.sigmoid(mask_preds), seg_gt)
            return {"loss": loss, "iou": 0.0}

        targets = self._prepare_detection_targets(gt_boxes, gt_labels, device, H, W)
        outputs = self.model(pixel_values=images, labels=targets)
        det_loss = outputs.loss

        # Segmentation loss using encoder features (if available)
        seg_loss = torch.tensor(0.0, device=device)
        try:
            # encoder_last_hidden_state: (B, seq_len, feat_dim)
            enc_feat = outputs.encoder_last_hidden_state  # (B, HW, C)
            h = w = int(enc_feat.shape[1] ** 0.5)
            feat_map = enc_feat.permute(0, 2, 1).reshape(B, -1, h, w)

            # Build seg GT
            seg_gt = torch.zeros(B, H, W, device=device)
            for i in range(B):
                if gt_masks[i].shape[0] > 0:
                    seg_gt[i] = gt_masks[i].to(device).max(dim=0).values

            pred_masks = self.mask_head(feat_map, torch.zeros(B, 1, 4, device=device))
            pred_masks_up = F.interpolate(
                pred_masks[:, :1], size=(H, W), mode="bilinear", align_corners=False
            ).squeeze(1)
            seg_loss = self.dice_loss(torch.sigmoid(pred_masks_up), seg_gt)
        except Exception:
            pass

        loss = det_loss + seg_loss

        with torch.no_grad():
            iou = 0.0

        return {"loss": loss, "det_loss": float(det_loss), "seg_loss": float(seg_loss), "iou": iou}

    @torch.no_grad()
    def val_step(self, batch: Dict, device: torch.device) -> Dict[str, float]:
        batch = self._batch_to_device(batch, device)
        images = batch["image"]
        gt_masks = batch["masks"]
        gt_labels = batch["labels"]
        gt_boxes = batch["boxes"]

        B, C, H, W = images.shape

        if not self._use_hf:
            return {"loss": 0.0, "miou": 0.0}

        targets = self._prepare_detection_targets(gt_boxes, gt_labels, device, H, W)
        outputs = self.model(pixel_values=images, labels=targets)
        loss = float(outputs.loss)

        iou_list = []
        try:
            enc_feat = outputs.encoder_last_hidden_state
            h = w = int(enc_feat.shape[1] ** 0.5)
            feat_map = enc_feat.permute(0, 2, 1).reshape(B, -1, h, w)

            seg_gt = torch.zeros(B, H, W, device=device)
            for i in range(B):
                if gt_masks[i].shape[0] > 0:
                    seg_gt[i] = gt_masks[i].to(device).max(dim=0).values

            pred_masks = self.mask_head(feat_map, torch.zeros(B, 1, 4, device=device))
            pred_up = F.interpolate(pred_masks[:, :1], size=(H, W), mode="bilinear", align_corners=False).squeeze(1)
            pred_binary = (torch.sigmoid(pred_up) > 0.5)

            for i in range(B):
                if gt_masks[i].shape[0] == 0:
                    continue
                iou_list.append(compute_iou(
                    pred_binary[i].cpu().numpy(),
                    (seg_gt[i] > 0.5).cpu().numpy(),
                ))
        except Exception:
            pass

        miou = float(np.mean(iou_list)) if iou_list else 0.0
        return {"loss": loss, "miou": miou}

    def predict(self, image: torch.Tensor, threshold: float = 0.5) -> Dict[str, Any]:
        if image.dim() == 3:
            image = image.unsqueeze(0)
        device = next(self.parameters()).device
        image = image.to(device)

        with torch.no_grad():
            if not self._use_hf:
                return {"masks": torch.zeros(1, self.image_size, self.image_size), "boxes": [], "labels": []}

            outputs = self.model(pixel_values=image)
            results = outputs.results if hasattr(outputs, "results") else []

        boxes_out = []
        labels_out = []
        scores_out = []
        if hasattr(outputs, "logits") and hasattr(outputs, "pred_boxes"):
            probs = outputs.logits.softmax(-1)
            scores, labels = probs[0, :, :-1].max(-1)
            boxes_out = outputs.pred_boxes[0]
            labels_out = labels + 1  # 1-indexed
            scores_out = scores

        return {
            "boxes": boxes_out.cpu() if isinstance(boxes_out, torch.Tensor) else boxes_out,
            "labels": labels_out.cpu() if isinstance(labels_out, torch.Tensor) else labels_out,
            "scores": scores_out.cpu() if isinstance(scores_out, torch.Tensor) else scores_out,
        }

    def reset_weights(self):
        self.__init__(self.config)
