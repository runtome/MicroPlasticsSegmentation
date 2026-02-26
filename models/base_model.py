"""
Abstract base model that all models must implement.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all segmentation models.

    Subclasses must implement:
        - forward(x) -> model output
        - train_step(batch, device) -> dict with 'loss' key
        - val_step(batch, device) -> dict with 'loss', metric keys
        - predict(image) -> dict with masks, labels, scores
        - reset_weights() -> reinitialize weights (for k-fold CV)
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.num_classes = config.get("num_classes", 3)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Any:
        pass

    @abstractmethod
    def train_step(self, batch: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Perform a single training step.
        Returns dict with at least 'loss' key (torch.Tensor, requires grad).
        """
        pass

    @abstractmethod
    def val_step(self, batch: Dict, device: torch.device) -> Dict[str, float]:
        """
        Perform a single validation step.
        Returns dict with 'loss' and metric keys (float values).
        """
        pass

    @abstractmethod
    def predict(self, image: torch.Tensor, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Run inference on a single image tensor (C, H, W).
        Returns dict with 'masks', 'labels', 'scores'.
        """
        pass

    def reset_weights(self):
        """Reinitialize all weights — used in k-fold CV."""
        for module in self.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _batch_to_device(self, batch: Dict, device: torch.device) -> Dict:
        """Move batch tensors to device."""
        result = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.to(device)
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                result[k] = [t.to(device) for t in v]
            else:
                result[k] = v
        return result
