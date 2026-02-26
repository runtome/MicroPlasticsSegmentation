"""
Evaluation metrics for instance segmentation.
Per-class IoU, mAP50, mAP75, F1, precision, recall.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def compute_iou(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Binary IoU between two masks (H, W)."""
    pred_bool = pred_mask.astype(bool)
    gt_bool = gt_mask.astype(bool)
    intersection = (pred_bool & gt_bool).sum()
    union = (pred_bool | gt_bool).sum()
    return float(intersection) / float(union + eps)


def compute_iou_batch(
    pred_masks: torch.Tensor,
    gt_masks: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute pairwise IoU between predicted and ground truth masks.

    Args:
        pred_masks: (N, H, W) binary
        gt_masks:   (M, H, W) binary

    Returns:
        iou_matrix: (N, M) float
    """
    N = pred_masks.shape[0]
    M = gt_masks.shape[0]

    pred_flat = pred_masks.view(N, -1).float()  # (N, H*W)
    gt_flat = gt_masks.view(M, -1).float()       # (M, H*W)

    intersection = torch.mm(pred_flat, gt_flat.t())  # (N, M)
    pred_areas = pred_flat.sum(dim=1, keepdim=True)   # (N, 1)
    gt_areas = gt_flat.sum(dim=1, keepdim=True)        # (M, 1)
    union = pred_areas + gt_areas.t() - intersection   # (N, M)

    return intersection / (union + eps)


def compute_ap(
    pred_scores: List[float],
    pred_matches: List[bool],
    n_gt: int,
    eps: float = 1e-6,
) -> float:
    """
    Compute Average Precision given ranked predictions.

    Args:
        pred_scores: confidence scores for each prediction (sorted descending)
        pred_matches: whether each prediction matches a GT (True/False)
        n_gt: total number of ground truth instances
    """
    if n_gt == 0:
        return 0.0

    tp = np.cumsum(pred_matches).astype(float)
    fp = np.cumsum([not m for m in pred_matches]).astype(float)

    precision = tp / (tp + fp + eps)
    recall = tp / n_gt

    # Add sentinel values
    recall = np.concatenate([[0.0], recall, [1.0]])
    precision = np.concatenate([[1.0], precision, [0.0]])

    # Make precision monotonically decreasing
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    # Compute area under PR curve
    idx = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[idx + 1] - recall[idx]) * precision[idx + 1])
    return float(ap)


def match_predictions_to_gt(
    pred_masks: np.ndarray,
    pred_scores: np.ndarray,
    pred_labels: np.ndarray,
    gt_masks: np.ndarray,
    gt_labels: np.ndarray,
    iou_threshold: float = 0.5,
) -> Dict[int, List]:
    """
    Match predictions to GT instances using IoU threshold, per class.

    Returns:
        dict mapping class_id -> list of (score, is_matched) tuples
    """
    class_ids = set(np.unique(gt_labels).tolist()) | set(np.unique(pred_labels).tolist())
    results = {int(c): [] for c in class_ids}
    n_gt_per_class = {int(c): int((gt_labels == c).sum()) for c in class_ids}

    gt_matched = np.zeros(len(gt_masks), dtype=bool)

    # Sort by descending score
    order = np.argsort(-pred_scores)
    pred_masks = pred_masks[order]
    pred_scores = pred_scores[order]
    pred_labels = pred_labels[order]

    for i, (pmask, pscore, plabel) in enumerate(
        zip(pred_masks, pred_scores, pred_labels)
    ):
        cls = int(plabel)
        # Find best matching GT of same class
        best_iou = 0.0
        best_j = -1
        for j, (gmask, glabel) in enumerate(zip(gt_masks, gt_labels)):
            if int(glabel) != cls or gt_matched[j]:
                continue
            iou = compute_iou(pmask, gmask)
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_iou >= iou_threshold and best_j >= 0:
            results[cls].append((pscore, True))
            gt_matched[best_j] = True
        else:
            results[cls].append((pscore, False))

    return results, n_gt_per_class


def compute_map(
    all_pred_masks: List[np.ndarray],
    all_pred_scores: List[np.ndarray],
    all_pred_labels: List[np.ndarray],
    all_gt_masks: List[np.ndarray],
    all_gt_labels: List[np.ndarray],
    iou_threshold: float = 0.5,
    class_ids: List[int] = None,
) -> Dict[str, float]:
    """
    Compute mAP across all images.

    Returns dict with 'mAP' and per-class 'AP_{class_id}'.
    """
    if class_ids is None:
        class_ids = [1, 2, 3]  # Fiber, Fragment, Film

    class_results = {c: [] for c in class_ids}
    n_gt_per_class = {c: 0 for c in class_ids}

    for pmasks, pscores, plabels, gmasks, glabels in zip(
        all_pred_masks, all_pred_scores, all_pred_labels, all_gt_masks, all_gt_labels
    ):
        if len(gmasks) == 0:
            continue
        pmasks_np = np.array(pmasks) if not isinstance(pmasks, np.ndarray) else pmasks
        pscores_np = np.array(pscores) if not isinstance(pscores, np.ndarray) else pscores
        plabels_np = np.array(plabels) if not isinstance(plabels, np.ndarray) else plabels
        gmasks_np = np.array(gmasks) if not isinstance(gmasks, np.ndarray) else gmasks
        glabels_np = np.array(glabels) if not isinstance(glabels, np.ndarray) else glabels

        if len(pmasks_np) == 0:
            for c in class_ids:
                n_gt_per_class[c] += int((glabels_np == c).sum())
            continue

        res, n_gt = match_predictions_to_gt(
            pmasks_np, pscores_np, plabels_np,
            gmasks_np, glabels_np,
            iou_threshold=iou_threshold,
        )
        for c in class_ids:
            class_results[c].extend(res.get(c, []))
            n_gt_per_class[c] += n_gt.get(c, 0)

    ap_per_class = {}
    for c in class_ids:
        items = class_results[c]
        if not items:
            ap_per_class[c] = 0.0
            continue
        items.sort(key=lambda x: -x[0])
        scores = [s for s, _ in items]
        matches = [m for _, m in items]
        ap_per_class[c] = compute_ap(scores, matches, n_gt_per_class[c])

    map_val = float(np.mean(list(ap_per_class.values()))) if ap_per_class else 0.0

    result = {"mAP": map_val}
    for c, ap in ap_per_class.items():
        result[f"AP_{c}"] = ap
    return result


def compute_f1(
    pred_labels: np.ndarray,
    gt_labels: np.ndarray,
    class_ids: List[int] = None,
) -> Dict[str, float]:
    """Compute per-class and macro F1."""
    if class_ids is None:
        class_ids = [1, 2, 3]

    results = {}
    f1_scores = []
    for c in class_ids:
        tp = int(((pred_labels == c) & (gt_labels == c)).sum())
        fp = int(((pred_labels == c) & (gt_labels != c)).sum())
        fn = int(((pred_labels != c) & (gt_labels == c)).sum())
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        results[f"F1_{c}"] = f1
        results[f"Precision_{c}"] = precision
        results[f"Recall_{c}"] = recall
        f1_scores.append(f1)

    results["F1_macro"] = float(np.mean(f1_scores))
    return results


class MetricTracker:
    """Accumulates metrics across batches for epoch-level averaging."""

    def __init__(self):
        self._sums: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}

    def update(self, metrics: Dict[str, float], n: int = 1):
        for k, v in metrics.items():
            self._sums[k] = self._sums.get(k, 0.0) + float(v) * n
            self._counts[k] = self._counts.get(k, 0) + n

    def compute(self) -> Dict[str, float]:
        return {
            k: self._sums[k] / self._counts[k]
            for k in self._sums
            if self._counts[k] > 0
        }

    def reset(self):
        self._sums.clear()
        self._counts.clear()
