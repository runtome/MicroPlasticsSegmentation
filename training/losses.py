"""
Loss functions for microplastics segmentation models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation masks."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: sigmoid predictions (B, ...) in [0, 1]
            target: binary ground truth (B, ...) in {0, 1}
        """
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1).float()
        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )
        return 1.0 - dice


class FocalLoss(nn.Module):
    """Focal loss for class imbalance (binary or multi-class)."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(pred, target.float(), reduction="none")
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal


class CombinedLoss(nn.Module):
    """
    Combined segmentation + classification loss (per paper).

    L_total = L_class(BCE) + L_seg(Dice)

    Used for U-Net family models that have both segmentation and classification heads.
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        cls_weight: float = 1.0,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.dice = DiceLoss(smooth=smooth)
        self.dice_weight = dice_weight
        self.cls_weight = cls_weight

    def forward(
        self,
        seg_pred: torch.Tensor,
        seg_gt: torch.Tensor,
        cls_pred: torch.Tensor = None,
        cls_gt: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            seg_pred: (B, 1, H, W) or (B, H, W) sigmoid predictions
            seg_gt:   (B, H, W) binary ground truth
            cls_pred: (B, num_classes) sigmoid predictions (optional)
            cls_gt:   (B, num_classes) one-hot ground truth (optional)
        """
        dice_loss = self.dice(seg_pred, seg_gt)

        if cls_pred is not None and cls_gt is not None:
            cls_loss = F.binary_cross_entropy(cls_pred, cls_gt.float())
            return self.dice_weight * dice_loss + self.cls_weight * cls_loss

        return dice_loss


class BCEDiceLoss(nn.Module):
    """BCE + Dice combined for segmentation."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice = DiceLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(pred, target.float())
        dice = self.dice(torch.sigmoid(pred), target)
        return self.bce_weight * bce + self.dice_weight * dice


class DetectionLoss(nn.Module):
    """
    Combined detection loss for Mask R-CNN style models.
    L_total = L_box(SmoothL1) + L_class(CE) + L_mask(BCE)

    Note: torchvision's MaskRCNN returns a dict of losses directly,
    so this class is provided for reference / custom detection models.
    """

    def __init__(
        self,
        box_weight: float = 1.0,
        cls_weight: float = 1.0,
        mask_weight: float = 1.0,
    ):
        super().__init__()
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.mask_weight = mask_weight

    def forward(self, loss_dict: dict) -> torch.Tensor:
        """Sum weighted losses from torchvision's model output dict."""
        total = torch.tensor(0.0)
        if "loss_box_reg" in loss_dict:
            total = total + self.box_weight * loss_dict["loss_box_reg"]
        if "loss_classifier" in loss_dict:
            total = total + self.cls_weight * loss_dict["loss_classifier"]
        if "loss_mask" in loss_dict:
            total = total + self.mask_weight * loss_dict["loss_mask"]
        if "loss_objectness" in loss_dict:
            total = total + loss_dict["loss_objectness"]
        if "loss_rpn_box_reg" in loss_dict:
            total = total + loss_dict["loss_rpn_box_reg"]
        return total


def build_loss(loss_name: str, **kwargs) -> nn.Module:
    """Factory function to build loss by name."""
    registry = {
        "combined": CombinedLoss,
        "dice": DiceLoss,
        "focal": FocalLoss,
        "bce_dice": BCEDiceLoss,
        "detection": DetectionLoss,
    }
    if loss_name not in registry:
        raise ValueError(f"Unknown loss: {loss_name}. Available: {list(registry.keys())}")
    return registry[loss_name](**kwargs)
