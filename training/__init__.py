from .losses import CombinedLoss, FocalLoss, DiceLoss, DetectionLoss
from .metrics import compute_iou, compute_ap, compute_f1, MetricTracker

__all__ = [
    "CombinedLoss", "FocalLoss", "DiceLoss", "DetectionLoss",
    "compute_iou", "compute_ap", "compute_f1", "MetricTracker",
    "Trainer",
]


def __getattr__(name):
    if name == "Trainer":
        from .trainer import Trainer
        return Trainer
    raise AttributeError(f"module 'training' has no attribute {name!r}")
