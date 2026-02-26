"""
YOLO segmentation wrapper using Ultralytics.
Trains 3 variants: s, m, x — compare performance vs. parameter count.
Weights are auto-downloaded by Ultralytics at first use.
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import numpy as np


class YOLO26Wrapper:
    """
    Wrapper for Ultralytics YOLO segmentation models.

    Note: Ultralytics models are not standard nn.Module in the same pattern —
    they manage their own training loop. This wrapper provides:
      - train(): delegates to ultralytics trainer
      - predict(): delegates to ultralytics predictor
      - val(): delegates to ultralytics validator
    """

    VARIANTS = {
        "s": "yolo11s-seg.pt",
        "m": "yolo11m-seg.pt",
        "x": "yolo11x-seg.pt",
    }

    def __init__(self, config: dict, variant: str = "s"):
        from ultralytics import YOLO

        self.config = config
        self.variant = variant
        model_cfg = config.get("model", config)
        self.num_classes = model_cfg.get("num_classes", 3)
        self.class_names = model_cfg.get("class_names", ["Fiber", "Fragment", "Film"])

        weights = self.VARIANTS.get(variant, f"yolo11{variant}-seg.pt")
        self.model = YOLO(weights)  # auto-downloads if not present
        self._last_results = None

    def train(
        self,
        data_yaml: str,
        epochs: int = 50,
        imgsz: int = 640,
        batch: int = 16,
        device: str = "0",
        project: str = "outputs/checkpoints/yolo26",
        name: str = None,
        **kwargs,
    ) -> dict:
        """Run Ultralytics training loop."""
        name = name or f"yolo11{self.variant}-seg"
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=project,
            name=name,
            save=True,
            plots=True,
            **kwargs,
        )
        self._last_results = results
        return results

    def val(self, data_yaml: str, imgsz: int = 640, device: str = "0") -> dict:
        """Run validation."""
        results = self.model.val(data=data_yaml, imgsz=imgsz, device=device)
        metrics = {
            "mAP50": float(results.box.map50) if hasattr(results, "box") else 0.0,
            "mAP75": float(results.box.map75) if hasattr(results, "box") else 0.0,
            "mAP50-95": float(results.box.map) if hasattr(results, "box") else 0.0,
        }
        if hasattr(results, "seg"):
            metrics["seg_mAP50"] = float(results.seg.map50)
            metrics["seg_mAP75"] = float(results.seg.map75)
        return metrics

    def predict(
        self,
        source,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str = "0",
    ):
        """Run inference."""
        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            retina_masks=True,
        )
        return results

    def load_checkpoint(self, checkpoint_path: str):
        """Load trained weights."""
        from ultralytics import YOLO
        self.model = YOLO(checkpoint_path)

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    @classmethod
    def train_all_variants(
        cls,
        config: dict,
        data_yaml: str,
        variants: list = None,
    ) -> Dict[str, Any]:
        """Train all 3 variants and return results dict."""
        if variants is None:
            variants = ["s", "m", "x"]

        results = {}
        for v in variants:
            print(f"\nTraining YOLO11{v}-seg...")
            wrapper = cls(config, variant=v)
            train_cfg = config.get("training", {})
            result = wrapper.train(
                data_yaml=data_yaml,
                epochs=train_cfg.get("num_epochs", 50),
                imgsz=train_cfg.get("imgsz", 640),
                batch=train_cfg.get("batch_size", 16),
                device=str(train_cfg.get("device", 0)),
            )
            results[f"yolo11{v}-seg"] = result

        return results
