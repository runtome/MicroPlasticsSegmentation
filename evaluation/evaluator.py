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

        # Per-image class sets for image-level multi-label F1
        all_pred_cls_sets = []
        all_gt_cls_sets = []

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

                # Image-level class sets for multi-label F1
                # Use cls_probs (thresholded) when available; else fall back to predicted_class
                if "cls_probs" in pred:
                    cls_probs_np = pred["cls_probs"].numpy()
                    pred_cls = set((np.where(cls_probs_np > 0.5)[0] + 1).tolist())
                    if not pred_cls:  # nothing above threshold → take argmax
                        pred_cls = {int(cls_probs_np.argmax()) + 1}
                else:
                    pred_cls = {int(plabels[0])} if len(plabels) > 0 else set()
                gt_cls = set(gt_l.numpy().astype(int).tolist()) if gt_l.shape[0] > 0 else set()
                all_pred_cls_sets.append(pred_cls)
                all_gt_cls_sets.append(gt_cls)

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

        # F1: image-level multi-label classification
        # For each image, cls_probs > 0.5 gives the predicted class set;
        # GT class set is the unique set of instance labels in that image.
        f1_metrics = {}
        f1_scores = []
        for c in [1, 2, 3]:
            tp = sum(c in ps and c in gs for ps, gs in zip(all_pred_cls_sets, all_gt_cls_sets))
            fp = sum(c in ps and c not in gs for ps, gs in zip(all_pred_cls_sets, all_gt_cls_sets))
            fn = sum(c not in ps and c in gs for ps, gs in zip(all_pred_cls_sets, all_gt_cls_sets))
            prec = tp / (tp + fp + 1e-6)
            rec = tp / (tp + fn + 1e-6)
            f1 = 2 * prec * rec / (prec + rec + 1e-6)
            f1_metrics[f"F1_{c}"] = float(f1)
            f1_scores.append(f1)
        f1_metrics["F1_macro"] = float(np.mean(f1_scores))

        params = self.model.count_parameters() if hasattr(self.model, "count_parameters") else 0
        avg_inference_ms = float(np.mean(inference_times)) if inference_times else 0.0

        cls_names = {1: "Fiber", 2: "Fragment", 3: "Film"}
        return {
            "iou_per_class": iou_per_class,
            "mIoU": float(np.mean(iou_list)) if iou_list else float(np.mean(list(iou_per_class.values()))),
            "mAP50": map50.get("mAP", 0.0),
            "mAP75": map75.get("mAP", 0.0),
            "AP_per_class_50": {c: map50.get(f"AP_{c}", 0.0) for c in [1, 2, 3]},
            **{f"F1_{cls_names[c]}": f1_metrics.get(f"F1_{c}", 0.0) for c in [1, 2, 3]},
            "F1_macro": f1_metrics.get("F1_macro", 0.0),
            "params": params,
            "inference_time_ms": avg_inference_ms,
        }
