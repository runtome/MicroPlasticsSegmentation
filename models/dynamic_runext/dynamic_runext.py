"""
Dynamic Residual U-Net Extended (Dynamic RUNext).
Key features vs plain U-Net:
  - Residual blocks in encoder/decoder
  - Dynamic convolution (input-conditioned kernel weights)
  - Pixel shuffle for sub-pixel upsampling
  - Dual output head (seg + classify)
"""
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel
from ..unet.blocks import ResidualBlock, OutConv
from training.losses import CombinedLoss
from training.metrics import compute_iou


class DynamicConv(nn.Module):
    """
    Dynamic convolution: K parallel kernels weighted by input-conditioned attention.
    Lightweight version: K=4 parallel convolutions.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, K: int = 4):
        super().__init__()
        self.K = K
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)
            for _ in range(K)
        ])
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, K),
            nn.Softmax(dim=1),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.attention(x)  # (B, K)
        out = sum(
            weights[:, k].view(-1, 1, 1, 1) * self.convs[k](x)
            for k in range(self.K)
        )
        return self.relu(self.bn(out))


class PixelShuffleUp(nn.Module):
    """Sub-pixel convolution upsampling (pixel shuffle)."""

    def __init__(self, in_channels: int, out_channels: int, scale: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * scale * scale, kernel_size=3, padding=1, bias=False)
        self.ps = nn.PixelShuffle(scale)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.ps(self.conv(x))))


class DynamicResDown(nn.Module):
    """MaxPool + Dynamic Conv + Residual."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.dynamic_conv = DynamicConv(in_channels, out_channels)
        self.residual = ResidualBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.dynamic_conv(x)
        return self.residual(x)


class DynamicResUp(nn.Module):
    """PixelShuffle + Residual with skip connection."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = PixelShuffleUp(in_channels, in_channels // 2)
        self.residual = ResidualBlock(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        return self.residual(x)


class DynamicRUNext(BaseModel):
    """
    Dynamic Residual U-Net Extended.
    Encoder: residual + dynamic conv blocks.
    Decoder: pixel shuffle upsample + residual blocks.
    Dual head: seg + classify.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        model_cfg = config.get("model", config)
        in_channels = model_cfg.get("in_channels", 3)
        num_classes = model_cfg.get("num_classes", 3)
        base_ch = model_cfg.get("base_channels", 64)
        depth = model_cfg.get("depth", 4)
        dropout_p = model_cfg.get("dropout", 0.3)

        features = [base_ch * (2 ** i) for i in range(depth)]

        # Stem
        self.stem = ResidualBlock(in_channels, features[0])

        # Encoder
        self.downs = nn.ModuleList()
        for i in range(len(features) - 1):
            self.downs.append(DynamicResDown(features[i], features[i + 1]))

        # Bottleneck
        bottleneck_ch = features[-1] * 2
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            DynamicConv(features[-1], bottleneck_ch),
            ResidualBlock(bottleneck_ch, bottleneck_ch),
            nn.Dropout2d(dropout_p),
        )

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

        # Decoder
        self.ups = nn.ModuleList()
        enc_channels = list(reversed(features))
        dec_in = bottleneck_ch
        for i, skip_ch in enumerate(enc_channels):
            self.ups.append(DynamicResUp(dec_in, skip_ch, skip_ch))
            dec_in = skip_ch

        self.outc = OutConv(features[0], 1)
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
        # Encoder
        enc_feats = []
        x = self.stem(x)
        enc_feats.append(x)
        for down in self.downs:
            x = down(x)
            enc_feats.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Classification
        cls_out = self.cls_head(x)

        # Decoder
        for up, skip in zip(self.ups, reversed(enc_feats)):
            x = up(x, skip)

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
