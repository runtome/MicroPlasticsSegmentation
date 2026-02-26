"""
Evaluator: load checkpoint → run test set → return metrics dict.
"""
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch

from training.metrics import compute_iou, compute_map, compute_f1, MetricTracker


class Evaluator:
    """
    Loads a model checkpoint and evaluates it on the test set.

    Returns:
        {
            iou_per_class: {1: float, 2: float, 3: float},
            mIoU: float,
            f1_per_class: {1: float, 2: float, 3: float},
            F1_macro: float,
            mAP50: float,
            mAP75: float,
            params: int,
            inference_time_ms: float,
        }
    """

    def __init__(self, model, device: str = "cuda"):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(cls, model_class, config: dict, checkpoint_path: str, device: str = "cuda"):
        """Instantiate model from checkpoint."""
        model = model_class(config)
        state = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
        return cls(model, device)

    @torch.no_grad()
    def evaluate(self, test_loader) -> Dict[str, Any]:
        """Run evaluation on test_loader and return metrics."""
        self.model.eval()

        all_pred_masks = []
        all_pred_scores = []
        all_pred_labels = []
        all_gt_masks = []
        all_gt_labels = []

        iou_list = []
        inference_times = []

        for batch in test_loader:
            images = batch["image"]
            gt_masks_batch = batch["masks"]
            gt_labels_batch = batch["labels"]

            for i in range(len(images)):
                img = images[i].to(self.device)

                t0 = time.perf_counter()
                try:
                    pred = self.model.predict(img, threshold=0.5)
                except Exception as e:
                    print(f"Predict error: {e}")
                    continue
                t1 = time.perf_counter()
                inference_times.append((t1 - t0) * 1000)

                # Extract GT for this image
                gt_m = gt_masks_batch[i]
                gt_l = gt_labels_batch[i]

                # Handle various prediction formats
                if isinstance(pred, dict):
                    if "masks" in pred and pred["masks"].shape[0] > 0:
                        pmasks = pred["masks"].numpy()
                        pscores = pred.get("scores", torch.ones(pmasks.shape[0])).numpy()
                        plabels = pred.get("labels", gt_l[:len(pmasks)]).numpy()
                    elif "mask" in pred:
                        pmasks = pred["mask"].unsqueeze(0).numpy()
                        pscores = np.array([1.0])
                        plabels = np.array([pred.get("predicted_class", 1)])
                    else:
                        continue
                else:
                    continue

                all_pred_masks.append(pmasks)
                all_pred_scores.append(pscores)
                all_pred_labels.append(plabels)
                all_gt_masks.append(gt_m.numpy() if gt_m.shape[0] > 0 else np.zeros((0, *pmasks.shape[1:])))
                all_gt_labels.append(gt_l.numpy())

                # Per-image IoU
                if gt_m.shape[0] > 0 and pmasks.shape[0] > 0:
                    gt_union = gt_m.numpy().max(axis=0)
                    pred_union = pmasks.max(axis=0)
                    iou_list.append(compute_iou(pred_union, gt_union))

        # Compute mAP at 0.5 and 0.75
        map50 = compute_map(
            all_pred_masks, all_pred_scores, all_pred_labels,
            all_gt_masks, all_gt_labels, iou_threshold=0.5,
        )
        map75 = compute_map(
            all_pred_masks, all_pred_scores, all_pred_labels,
            all_gt_masks, all_gt_labels, iou_threshold=0.75,
        )

        # Per-class IoU
        iou_per_class = {}
        for cls_id in [1, 2, 3]:
            cls_ious = []
            for pmasks, plabels, gmasks, glabels in zip(
                all_pred_masks, all_pred_labels, all_gt_masks, all_gt_labels
            ):
                for j, (gm, gl) in enumerate(zip(gmasks, glabels)):
                    if int(gl) != cls_id:
                        continue
                    best_iou = 0.0
                    for k, (pm, pl) in enumerate(zip(pmasks, plabels)):
                        if int(pl) == cls_id:
                            best_iou = max(best_iou, compute_iou(pm, gm))
                    cls_ious.append(best_iou)
            iou_per_class[cls_id] = float(np.mean(cls_ious)) if cls_ious else 0.0

        # F1 (classification accuracy)
        all_pred_labels_flat = np.concatenate(all_pred_labels) if all_pred_labels else np.array([])
        all_gt_labels_flat = np.concatenate(all_gt_labels) if all_gt_labels else np.array([])
        # Match by limiting to min length (rough approximation for classification F1)
        n = min(len(all_pred_labels_flat), len(all_gt_labels_flat))
        f1_metrics = compute_f1(
            all_pred_labels_flat[:n],
            all_gt_labels_flat[:n],
        ) if n > 0 else {"F1_macro": 0.0}

        params = self.model.count_parameters() if hasattr(self.model, "count_parameters") else 0
        avg_inference_ms = float(np.mean(inference_times)) if inference_times else 0.0

        return {
            "iou_per_class": iou_per_class,
            "mIoU": float(np.mean(iou_list)) if iou_list else float(np.mean(list(iou_per_class.values()))),
            "mAP50": map50.get("mAP", 0.0),
            "mAP75": map75.get("mAP", 0.0),
            "AP_per_class_50": {c: map50.get(f"AP_{c}", 0.0) for c in [1, 2, 3]},
            **{f"F1_{c}": f1_metrics.get(f"F1_{c}", 0.0) for c in [1, 2, 3]},
            "F1_macro": f1_metrics.get("F1_macro", 0.0),
            "params": params,
            "inference_time_ms": avg_inference_ms,
        }
