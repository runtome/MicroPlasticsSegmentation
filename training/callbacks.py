"""
Training callbacks: EarlyStopping, ReduceLROnPlateau wrapper, ModelCheckpoint.
"""
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class EarlyStopping:
    """Stop training when a monitored metric stops improving."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "min",
        verbose: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        improved = (
            value < self.best_value - self.min_delta
            if self.mode == "min"
            else value > self.best_value + self.min_delta
        )
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: no improvement for {self.counter}/{self.patience} epochs")
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print("EarlyStopping triggered.")
        return self.should_stop


class ModelCheckpoint:
    """Save model checkpoint when monitored metric improves."""

    def __init__(
        self,
        checkpoint_dir: str,
        model_name: str,
        monitor: str = "val_miou",
        mode: str = "max",
        save_top_k: int = 1,
        verbose: bool = True,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.verbose = verbose
        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.best_path: Optional[Path] = None

    def __call__(
        self,
        model: nn.Module,
        epoch: int,
        metrics: dict,
        optimizer=None,
        fold: Optional[int] = None,
    ) -> Optional[str]:
        value = metrics.get(self.monitor)
        if value is None:
            return None

        improved = (
            value > self.best_value if self.mode == "max" else value < self.best_value
        )
        if improved:
            self.best_value = value
            fold_str = f"_fold{fold}" if fold is not None else ""
            metric_str = f"{value:.4f}".replace(".", "")
            filename = f"{self.model_name}{fold_str}_epoch{epoch:03d}_{self.monitor}{metric_str}.pth"
            save_path = self.checkpoint_dir / filename

            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "metrics": metrics,
                "best_value": self.best_value,
            }
            if optimizer is not None:
                state["optimizer_state_dict"] = optimizer.state_dict()

            torch.save(state, save_path)

            # Remove previous best if save_top_k == 1
            if self.save_top_k == 1 and self.best_path is not None and self.best_path.exists():
                self.best_path.unlink()

            self.best_path = save_path

            if self.verbose:
                print(f"Checkpoint saved: {save_path} ({self.monitor}={value:.4f})")

            return str(save_path)
        return None
