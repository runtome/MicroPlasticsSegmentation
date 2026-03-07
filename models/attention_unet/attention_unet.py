"""
Attention U-Net: U-Net with attention gates in skip connections.
Same dual-head design as U-Net (seg + classify).

Two build modes controlled by config:

  Mode A — Custom (default, no pretrained):
      encoder: null   OR   pretrained: false
      Builds from scratch using DoubleConv/Down blocks + AttentionUp decoder.

  Mode B — Pretrained encoder via segmentation-models-pytorch:
      encoder: resnet34   (or any smp-supported encoder)
      pretrained: true    → loads ImageNet weights
      pretrained: false   → smp encoder, random init
      Decoder: custom AttentionUp blocks on top of smp encoder features.
      Classification head taps the encoder bottleneck features.
      Requires: pip install segmentation-models-pytorch

Both modes expose identical forward / train_step / val_step / predict interface.
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

    Config keys read:
        model.encoder    : encoder name (e.g. "resnet34") or null
        model.pretrained : true  → ImageNet weights via smp
                           false → random init (custom or smp)
        model.features   : [64, 128, 256, 512]  (custom mode only)
        model.in_channels: 3
        model.num_classes: 2
        model.dropout    : 0.3
    """

    def __init__(self, config: dict):
        super().__init__(config)
        model_cfg = config.get("model", config)
        self._in_channels = model_cfg.get("in_channels", 3)
        self._num_classes = model_cfg.get("num_classes", 2)
        self._features = model_cfg.get("features", [64, 128, 256, 512])
        self._dropout_p = model_cfg.get("dropout", 0.3)
        self._encoder_name = model_cfg.get("encoder", None)
        self._pretrained = model_cfg.get("pretrained", False)

        self.loss_fn = CombinedLoss()

        if self._encoder_name and self._encoder_name.lower() not in ("null", "none", ""):
            self._use_smp = True
            self._build_smp()
        else:
            self._use_smp = False
            self._build_custom()

    # ── Build helpers ─────────────────────────────────────────────────────────

    def _build_custom(self):
        """Custom Attention U-Net built from scratch."""
        in_ch = self._in_channels
        num_cls = self._num_classes
        features = self._features
        dropout_p = self._dropout_p

        # Encoder
        self.inc = DoubleConv(in_ch, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])

        bottleneck_ch = features[3]
        self.down4 = Down(features[3], bottleneck_ch)
        self.bottleneck_dropout = nn.Dropout2d(dropout_p)

        # Decoder with attention
        self.up1 = AttentionUp(bottleneck_ch, features[3], features[3], F_int=features[3] // 2)
        self.up2 = AttentionUp(features[3], features[2], features[2], F_int=features[2] // 2)
        self.up3 = AttentionUp(features[2], features[1], features[1], F_int=features[1] // 2)
        self.up4 = AttentionUp(features[1], features[0], features[0], F_int=features[0] // 2)
        self.outc = OutConv(features[0], 1)

        # Classification head
        self.cls_head = self._make_cls_head(bottleneck_ch, num_cls, dropout_p)
        self._init_weights()

    def _build_smp(self):
        """Attention U-Net with pretrained encoder via segmentation-models-pytorch."""
        try:
            import segmentation_models_pytorch as smp
        except ImportError:
            raise ImportError(
                "segmentation-models-pytorch is required for pretrained encoders.\n"
                "Install with: pip install segmentation-models-pytorch"
            )

        encoder_weights = "imagenet" if self._pretrained else None
        print(f"[AttentionUNet] Using smp encoder='{self._encoder_name}'  "
              f"weights='{encoder_weights}'")

        # Use smp to get encoder only
        self._smp_helper = smp.Unet(
            encoder_name=self._encoder_name,
            encoder_weights=encoder_weights,
            in_channels=self._in_channels,
            classes=1,
            activation=None,
        )
        self.encoder = self._smp_helper.encoder

        # smp encoder.out_channels: e.g. [3, 64, 64, 128, 256, 512] for resnet34
        enc_channels = list(self.encoder.out_channels)
        print(f"[AttentionUNet] Encoder channels: {enc_channels}")

        # Build attention decoder on encoder features
        # enc_channels indices: 0=input, 1..N=stages
        # We decode from the deepest (last) back to shallowest
        # Skip connections come from stages [N-1, N-2, ..., 1]
        # Some encoders (e.g. mit_b2) have 0-channel stages — skip those
        n = len(enc_channels) - 1  # number of encoder stages (excl. input)
        bottleneck_ch = enc_channels[n]

        self.bottleneck_dropout = nn.Dropout2d(self._dropout_p)

        # Collect valid skip indices (non-zero channels, excluding bottleneck)
        self._skip_indices = [i for i in range(n - 1, 0, -1) if enc_channels[i] > 0]

        # Build decoder blocks: from bottleneck upward
        # Each block: AttentionUp(in_ch=prev_decoder_out, skip_ch=enc_channels[i], out_ch)
        self.attn_ups = nn.ModuleList()
        dec_ch = bottleneck_ch
        for i in self._skip_indices:
            skip_ch = enc_channels[i]
            out_ch = skip_ch  # match skip channel width
            f_int = max(skip_ch // 2, 16)
            self.attn_ups.append(AttentionUp(dec_ch, skip_ch, out_ch, F_int=f_int))
            dec_ch = out_ch

        self.outc = OutConv(dec_ch, 1)

        # Classification head from bottleneck
        self.cls_head = self._make_cls_head(bottleneck_ch, self._num_classes, self._dropout_p)

        # Only init decoder + cls head — leave pretrained encoder intact
        self._init_decoder_and_cls_head()

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

    def _init_decoder_and_cls_head(self):
        """Init decoder + cls head only (smp mode — preserve encoder weights)."""
        for module in [self.attn_ups, self.outc, self.cls_head]:
            for m in module.modules():
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

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        seg_out = self.outc(x)

        return seg_out, cls_out

    def _forward_smp(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encoder: returns list of feature maps [input, stage1, ..., stageN]
        enc_features = self.encoder(x)
        bottleneck = enc_features[-1]
        bottleneck = self.bottleneck_dropout(bottleneck)

        cls_out = self.cls_head(bottleneck)

        # Decoder with attention gates on skip connections
        d = bottleneck
        for skip_idx, attn_up in zip(self._skip_indices, self.attn_ups):
            d = attn_up(d, enc_features[skip_idx])

        seg_out = self.outc(d)

        # smp encoders may downsample at stage 1 (e.g. resnet stem stride 2),
        # so output may be smaller than input — upsample to match
        if seg_out.shape[2:] != x.shape[2:]:
            seg_out = F.interpolate(seg_out, size=x.shape[2:], mode="bilinear", align_corners=True)

        return seg_out, cls_out

    # ── Train / Val steps ─────────────────────────────────────────────────────

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

    # ── K-fold CV reset ───────────────────────────────────────────────────────

    def reset_weights(self):
        """
        Re-initialise weights between CV folds.
        - Custom mode : Kaiming/Xavier reinit (same as __init__)
        - SMP mode    : reload pretrained encoder weights + reinit decoder & cls head
        """
        if self._use_smp:
            self._build_smp()
        else:
            self._init_weights()
