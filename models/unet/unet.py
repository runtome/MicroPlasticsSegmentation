"""
U-Net with dual output head (segmentation + classification).

Two build modes controlled by config:

  Mode A — Custom (default, no pretrained):
      encoder: null   OR   pretrained: false
      Builds from scratch using DoubleConv/Down/Up blocks.
      Weights: Kaiming normal (Conv) + Xavier (Linear).

  Mode B — Pretrained encoder via segmentation-models-pytorch:
      encoder: resnet34   (or any smp-supported encoder)
      pretrained: true    → loads ImageNet weights
      pretrained: false   → smp encoder, random init
      Decoder: smp U-Net decoder.
      Classification head taps the encoder bottleneck features.
      Requires: pip install segmentation-models-pytorch

Both modes expose identical forward / train_step / val_step / predict interface.
"""
from typing import Dict, Any, Tuple, Optional

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
    - Segmentation head : (B, 1, H, W)  — binary mask logits
    - Classification head: (B, num_classes) — multi-label sigmoid

    Config keys read:
        model.encoder    : encoder name (e.g. "resnet34") or null
        model.pretrained : true  → ImageNet weights via smp
                           false → random init (custom or smp)
        model.features   : [64, 128, 256, 512]  (custom mode only)
        model.in_channels: 3
        model.num_classes: 3
        model.dropout    : 0.3
    """

    def __init__(self, config: dict):
        super().__init__(config)
        model_cfg = config.get("model", config)
        self._in_channels = model_cfg.get("in_channels", 3)
        self._num_classes  = model_cfg.get("num_classes", 3)
        self._features     = model_cfg.get("features", [64, 128, 256, 512])
        self._dropout_p    = model_cfg.get("dropout", 0.3)
        self._encoder_name = model_cfg.get("encoder", None)
        self._pretrained   = model_cfg.get("pretrained", False)

        self.loss_fn = CombinedLoss()

        if self._encoder_name and self._encoder_name.lower() not in ("null", "none", ""):
            self._use_smp = True
            self._build_smp()
        else:
            self._use_smp = False
            self._build_custom()

    # ── Build helpers ─────────────────────────────────────────────────────────

    def _build_custom(self):
        """Custom U-Net built from scratch (original implementation)."""
        in_ch     = self._in_channels
        num_cls   = self._num_classes
        features  = self._features
        dropout_p = self._dropout_p
        bilinear  = True
        factor    = 2 if bilinear else 1

        self.inc   = DoubleConv(in_ch, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])

        bottleneck_ch = features[3] * 2 // factor
        self.down4 = Down(features[3], bottleneck_ch)
        self.bottleneck_dropout = nn.Dropout2d(dropout_p)

        self.up1  = Up(features[3] * 2, features[3] // factor, bilinear)
        self.up2  = Up(features[3],     features[2] // factor, bilinear)
        self.up3  = Up(features[2],     features[1] // factor, bilinear)
        self.up4  = Up(features[1],     features[0],           bilinear)
        self.outc = OutConv(features[0], 1)

        self.cls_head = self._make_cls_head(bottleneck_ch, num_cls, dropout_p)
        self._init_weights()

    def _build_smp(self):
        """U-Net with pretrained encoder via segmentation-models-pytorch."""
        try:
            import segmentation_models_pytorch as smp
        except ImportError:
            raise ImportError(
                "segmentation-models-pytorch is required for pretrained encoders.\n"
                "Install with: pip install segmentation-models-pytorch"
            )

        encoder_weights = "imagenet" if self._pretrained else None
        print(f"[UNet] Using smp encoder='{self._encoder_name}'  "
              f"weights='{encoder_weights}'")

        self.smp_model = smp.Unet(
            encoder_name=self._encoder_name,
            encoder_weights=encoder_weights,
            in_channels=self._in_channels,
            classes=1,          # binary seg output (logits)
            activation=None,    # raw logits — sigmoid applied later
        )

        # Bottleneck channel count from the encoder
        bottleneck_ch = self.smp_model.encoder.out_channels[-1]
        print(f"[UNet] Bottleneck channels: {bottleneck_ch}")

        self.cls_head = self._make_cls_head(
            bottleneck_ch, self._num_classes, self._dropout_p
        )

        # Only init the classification head — leave pretrained encoder intact
        self._init_cls_head_only()

    @staticmethod
    def _make_cls_head(in_ch: int, num_cls: int, dropout_p: float) -> nn.Sequential:
        """GlobalAvgPool → FC(256) → ReLU → FC(num_cls) → Sigmoid."""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_ch, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(256, num_cls),
            nn.Sigmoid(),
        )

    # ── Weight initialisation ─────────────────────────────────────────────────

    def _init_weights(self):
        """Init all weights (custom mode — no pretrained encoder)."""
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

    def _init_cls_head_only(self):
        """Init only the classification head (smp mode — preserve encoder weights)."""
        for m in self.cls_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            seg_out : (B, 1, H, W) — binary segmentation logits
            cls_out : (B, num_classes) — class probabilities
        """
        if self._use_smp:
            return self._forward_smp(x)
        return self._forward_custom(x)

    def _forward_custom(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.bottleneck_dropout(x5)

        cls_out = self.cls_head(x5)

        x = self.up1(x5, x4)
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)
        seg_out = self.outc(x)

        return seg_out, cls_out

    def _forward_smp(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        enc_features = self.smp_model.encoder(x)       # list of feature maps
        # smp API differs by version: try list first, fall back to unpacked
        try:
            decoder_out = self.smp_model.decoder(enc_features)
        except TypeError:
            decoder_out = self.smp_model.decoder(*enc_features)
        seg_out  = self.smp_model.segmentation_head(decoder_out)  # (B,1,H,W)
        cls_out  = self.cls_head(enc_features[-1])                 # bottleneck
        return seg_out, cls_out

    # ── Train / Val steps ─────────────────────────────────────────────────────

    def train_step(self, batch: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
        batch  = self._batch_to_device(batch, device)
        images = batch["image"]
        gt_masks  = batch["masks"]
        gt_labels = batch["labels"]

        B = images.shape[0]
        seg_pred, cls_pred = self(images)
        seg_pred_sig = torch.sigmoid(seg_pred.squeeze(1))  # (B, H, W)

        H, W = images.shape[2], images.shape[3]
        seg_gt = torch.zeros(B, H, W, device=device)
        cls_gt = torch.zeros(B, self.num_classes, device=device)

        for i in range(B):
            if gt_masks[i].shape[0] > 0:
                seg_gt[i] = gt_masks[i].to(device).max(dim=0).values
                for lbl in gt_labels[i]:
                    cls_gt[i, lbl.item() - 1] = 1.0  # 1-indexed → 0-indexed

        loss = self.loss_fn(seg_pred_sig, seg_gt, cls_pred, cls_gt)

        with torch.no_grad():
            pred_bin = (seg_pred_sig > 0.5).float()
            iou = compute_iou(
                pred_bin.cpu().numpy().reshape(-1),
                seg_gt.cpu().numpy().reshape(-1),
            )

        return {"loss": loss, "iou": iou}

    def val_step(self, batch: Dict, device: torch.device) -> Dict[str, float]:
        batch  = self._batch_to_device(batch, device)
        images = batch["image"]
        gt_masks  = batch["masks"]
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
        seg_mask  = (torch.sigmoid(seg_pred.squeeze(1)) > threshold).squeeze(0)
        cls_probs = cls_pred.squeeze(0)

        return {
            "mask": seg_mask.cpu(),
            "cls_probs": cls_probs.cpu(),
            "predicted_class": cls_probs.argmax().item() + 1,  # back to 1-indexed
        }

    # ── K-fold CV reset ───────────────────────────────────────────────────────

    def reset_weights(self):
        """
        Re-initialise weights between CV folds.
        - Custom mode : Kaiming/Xavier reinit (same as __init__)
        - SMP mode    : reload pretrained encoder weights + reinit cls head
        """
        if self._use_smp:
            # Re-instantiate the smp model to reload pretrained weights cleanly
            self._build_smp()
        else:
            self._init_weights()
