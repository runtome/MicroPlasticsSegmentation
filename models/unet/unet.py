"""
U-Net with dual output head (segmentation + classification).
Per the paper: parallel classification branch after bottleneck.
"""
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel
from .blocks import DoubleConv, Down, Up, OutConv
from training.losses import CombinedLoss
from training.metrics import compute_iou


class UNet(BaseModel):
    """
    U-Net encoder-decoder with:
    - Segmentation head: (B, 1, H, W) per instance — binary mask
    - Classification head: (B, num_classes) — multi-label sigmoid

    The classification head is a parallel branch from the bottleneck.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        model_cfg = config.get("model", config)
        in_channels = model_cfg.get("in_channels", 3)
        num_classes = model_cfg.get("num_classes", 3)
        features = model_cfg.get("features", [64, 128, 256, 512])
        bilinear = True
        dropout_p = model_cfg.get("dropout", 0.3)

        self.inc = DoubleConv(in_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])

        # Bottleneck
        # Standard bilinear U-Net: bottleneck output = features[-1]*2 // factor
        # So cat(skip=features[3], up(bottleneck))=features[3]*2 = Up's in_channels
        factor = 2 if bilinear else 1
        bottleneck_ch = features[3] * 2 // factor  # output channels of bottleneck
        self.down4 = Down(features[3], bottleneck_ch)  # x5 has bottleneck_ch channels
        self.bottleneck_dropout = nn.Dropout2d(dropout_p)

        # Decoder: Up(cat_channels, out_channels) where cat = skip + upsampled
        self.up1 = Up(features[3] * 2, features[3] // factor, bilinear)
        self.up2 = Up(features[3], features[2] // factor, bilinear)
        self.up3 = Up(features[2], features[1] // factor, bilinear)
        self.up4 = Up(features[1], features[0], bilinear)
        self.outc = OutConv(features[0], 1)  # Binary mask per instance

        # Classification head (parallel branch from bottleneck)
        # GlobalAvgPool → FC(256) → ReLU → FC(num_classes) → Sigmoid
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(bottleneck_ch, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(256, num_classes),
            nn.Sigmoid(),
        )

        self.loss_fn = CombinedLoss()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            seg_out: (B, 1, H, W) — binary segmentation logits
            cls_out: (B, num_classes) — classification probabilities
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.bottleneck_dropout(x5)

        # Classification branch
        cls_out = self.cls_head(x5)

        # Decoder branch
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        seg_out = self.outc(x)

        return seg_out, cls_out

    def train_step(self, batch: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
        batch = self._batch_to_device(batch, device)
        images = batch["image"]          # (B, C, H, W)
        gt_masks = batch["masks"]        # list of (N_i, H, W)
        gt_labels = batch["labels"]      # list of (N_i,)

        B = images.shape[0]
        seg_pred, cls_pred = self(images)
        seg_pred_sig = torch.sigmoid(seg_pred.squeeze(1))  # (B, H, W)

        # Build aggregated binary segmentation GT (union of all instance masks)
        H, W = images.shape[2], images.shape[3]
        seg_gt = torch.zeros(B, H, W, device=device)
        cls_gt = torch.zeros(B, self.num_classes, device=device)

        for i in range(B):
            if gt_masks[i].shape[0] > 0:
                seg_gt[i] = gt_masks[i].to(device).max(dim=0).values
                for lbl in gt_labels[i]:
                    cls_gt[i, lbl.item() - 1] = 1.0  # 1-indexed → 0-indexed

        loss = self.loss_fn(seg_pred_sig, seg_gt, cls_pred, cls_gt)

        # Compute batch IoU
        with torch.no_grad():
            pred_bin = (seg_pred_sig > 0.5).float()
            iou = compute_iou(
                pred_bin.cpu().numpy().reshape(-1),
                seg_gt.cpu().numpy().reshape(-1),
            )

        return {"loss": loss, "iou": iou}

    def val_step(self, batch: Dict, device: torch.device) -> Dict[str, float]:
        batch = self._batch_to_device(batch, device)
        images = batch["image"]
        gt_masks = batch["masks"]
        gt_labels = batch["labels"]

        B = images.shape[0]
        seg_pred, cls_pred = self(images)
        seg_pred_sig = torch.sigmoid(seg_pred.squeeze(1))

        H, W = images.shape[2], images.shape[3]
        seg_gt = torch.zeros(B, H, W, device=device)
        cls_gt = torch.zeros(B, self.num_classes, device=device)

        for i in range(B):
            if gt_masks[i].shape[0] > 0:
                seg_gt[i] = gt_masks[i].to(device).max(dim=0).values
                for lbl in gt_labels[i]:
                    cls_gt[i, lbl.item() - 1] = 1.0

        loss = self.loss_fn(seg_pred_sig, seg_gt, cls_pred, cls_gt)

        pred_bin = (seg_pred_sig > 0.5).float()
        iou = compute_iou(
            pred_bin.cpu().numpy().reshape(-1),
            seg_gt.cpu().numpy().reshape(-1),
        )

        return {"loss": float(loss), "miou": iou}

    def predict(self, image: torch.Tensor, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Args:
            image: (C, H, W) or (1, C, H, W) tensor
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        seg_pred, cls_pred = self(image)
        seg_mask = (torch.sigmoid(seg_pred.squeeze(1)) > threshold).squeeze(0)
        cls_probs = cls_pred.squeeze(0)
        predicted_class = cls_probs.argmax().item() + 1  # back to 1-indexed

        return {
            "mask": seg_mask.cpu(),
            "cls_probs": cls_probs.cpu(),
            "predicted_class": predicted_class,
        }
