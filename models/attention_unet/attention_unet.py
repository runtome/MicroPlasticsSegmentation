"""
Attention U-Net: U-Net with attention gates in skip connections.
Same dual-head design as U-Net (seg + classify).
"""
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel
from ..unet.blocks import DoubleConv, Down, OutConv
from .attention_gate import AttentionGate
from training.losses import CombinedLoss
from training.metrics import compute_iou


class AttentionUp(nn.Module):
    """Upsample + AttentionGate on skip + DoubleConv."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, F_int: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.attn = AttentionGate(F_g=in_channels, F_l=skip_channels, F_int=F_int)
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        skip = self.attn(g=x, x=skip)
        # Pad if needed
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class AttentionUNet(BaseModel):
    """
    Attention U-Net with classification head.
    Same architecture as U-Net but skip connections pass through attention gates.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        model_cfg = config.get("model", config)
        in_channels = model_cfg.get("in_channels", 3)
        num_classes = model_cfg.get("num_classes", 3)
        features = model_cfg.get("features", [64, 128, 256, 512])
        dropout_p = model_cfg.get("dropout", 0.3)

        # Encoder
        self.inc = DoubleConv(in_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])

        # bottleneck outputs features[3] channels (halved by bilinear factor=2)
        bottleneck_ch = features[3]  # output of down4
        self.down4 = Down(features[3], bottleneck_ch)
        self.bottleneck_dropout = nn.Dropout2d(dropout_p)

        # Decoder with attention
        # AttentionUp(dec_in, skip_ch, out_ch): dec_in=bottleneck, skip=features[i]
        self.up1 = AttentionUp(bottleneck_ch, features[3], features[3], F_int=features[3] // 2)
        self.up2 = AttentionUp(features[3], features[2], features[2], F_int=features[2] // 2)
        self.up3 = AttentionUp(features[2], features[1], features[1], F_int=features[1] // 2)
        self.up4 = AttentionUp(features[1], features[0], features[0], F_int=features[0] // 2)
        self.outc = OutConv(features[0], 1)

        # Classification head
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
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.bottleneck_dropout(x5)

        cls_out = self.cls_head(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        seg_out = self.outc(x)

        return seg_out, cls_out

    def train_step(self, batch: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
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

        with torch.no_grad():
            pred_bin = (seg_pred_sig > 0.5).float()
            iou = compute_iou(pred_bin.cpu().numpy().reshape(-1), seg_gt.cpu().numpy().reshape(-1))

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
        iou = compute_iou(pred_bin.cpu().numpy().reshape(-1), seg_gt.cpu().numpy().reshape(-1))

        return {"loss": float(loss), "miou": iou}

    def predict(self, image: torch.Tensor, threshold: float = 0.5) -> Dict[str, Any]:
        if image.dim() == 3:
            image = image.unsqueeze(0)
        seg_pred, cls_pred = self(image)
        seg_mask = (torch.sigmoid(seg_pred.squeeze(1)) > threshold).squeeze(0)
        cls_probs = cls_pred.squeeze(0)
        return {
            "mask": seg_mask.cpu(),
            "cls_probs": cls_probs.cpu(),
            "predicted_class": cls_probs.argmax().item() + 1,
        }
