"""
Build comparison table across all models → CSV.
"""
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np


# Metric display names
METRIC_COLS = [
    "model",
    "mIoU",
    "IoU_Fiber",
    "IoU_Fragment",
    "IoU_Film",
    "mAP50",
    "mAP75",
    "F1_macro",
    "F1_Fiber",
    "F1_Fragment",
    "F1_Film",
    "params_M",
    "inference_time_ms",
]

CLASS_NAMES = {1: "Fiber", 2: "Fragment", 3: "Film"}


class ModelComparison:
    """
    Aggregate evaluation results from multiple models into a comparison DataFrame.
    """

    def __init__(self):
        self.records = []

    def add_model(self, model_name: str, metrics: Dict[str, Any]):
        """
        Add a model's evaluation results.

        Args:
            model_name: display name for the model
            metrics: dict from Evaluator.evaluate()
        """
        # iou_per_class keys may be int (from evaluator) or str (after JSON round-trip)
        iou_per_class = metrics.get("iou_per_class", {})
        def _iou(cls_id):
            return float(iou_per_class.get(cls_id, iou_per_class.get(str(cls_id), 0.0)))

        record = {
            "model": model_name,
            "mIoU": round(metrics.get("mIoU", 0.0), 4),
            "IoU_Fiber": round(_iou(1), 4),
            "IoU_Fragment": round(_iou(2), 4),
            "IoU_Film": round(_iou(3), 4),
            "mAP50": round(metrics.get("mAP50", 0.0), 4),
            "mAP75": round(metrics.get("mAP75", 0.0), 4),
            "F1_macro": round(metrics.get("F1_macro", 0.0), 4),
            "F1_Fiber": round(metrics.get("F1_Fiber", 0.0), 4),
            "F1_Fragment": round(metrics.get("F1_Fragment", 0.0), 4),
            "F1_Film": round(metrics.get("F1_Film", 0.0), 4),
            "params_M": round(metrics.get("params", 0) / 1e6, 2),
            "inference_time_ms": round(metrics.get("inference_time_ms", 0.0), 2),
        }
        self.records.append(record)

    def build_dataframe(self) -> pd.DataFrame:
        """Build comparison DataFrame sorted by mIoU descending."""
        if not self.records:
            return pd.DataFrame(columns=METRIC_COLS)
        df = pd.DataFrame(self.records)
        # Add missing columns
        for col in METRIC_COLS:
            if col not in df.columns:
                df[col] = 0.0
        df = df[METRIC_COLS].sort_values("mIoU", ascending=False).reset_index(drop=True)
        df.index = df.index + 1  # 1-indexed rank
        return df

    def save_csv(self, output_path: str) -> pd.DataFrame:
        """Save comparison table to CSV and return DataFrame."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df = self.build_dataframe()
        df.to_csv(output_path)
        print(f"\nModel comparison saved to: {output_path}")
        print(df.to_string())
        return df

    def print_table(self):
        """Print formatted comparison table."""
        df = self.build_dataframe()
        if df.empty:
            print("No models evaluated yet.")
            return
        print("\n" + "=" * 120)
        print("MODEL COMPARISON")
        print("=" * 120)
        print(df.to_string())
        print("=" * 120)


def load_results_from_dir(results_dir: str) -> "ModelComparison":
    """
    Load existing JSON result files from outputs/results/ and build comparison.
    """
    import json
    comparison = ModelComparison()
    results_dir = Path(results_dir)

    for json_file in sorted(results_dir.glob("*.json")):
        model_name = json_file.stem
        with open(json_file) as f:
            metrics = json.load(f)
        comparison.add_model(model_name, metrics)

    return comparison
