"""
Visualization utilities: draw masks + class labels on images.
"""
from typing import Dict, Any, Optional, List

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba


CLASS_NAMES = {0: "Background", 1: "Fiber", 2: "Fragment", 3: "Film"}

# Color palette per class (RGB)
CLASS_COLORS = {
    0: (128, 128, 128),  # Background — gray
    1: (255, 100, 100),  # Fiber — red
    2: (100, 200, 100),  # Fragment — green
    3: (100, 100, 255),  # Film — blue
}


def draw_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple,
    alpha: float = 0.4,
) -> np.ndarray:
    """Overlay a binary mask on an image with given color and transparency."""
    overlay = image.copy()
    mask_bool = mask.astype(bool)
    overlay[mask_bool] = (
        np.array(color) * alpha + image[mask_bool] * (1 - alpha)
    ).astype(np.uint8)
    return overlay


def draw_contour(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple,
    thickness: int = 2,
) -> np.ndarray:
    """Draw contour of a binary mask on an image."""
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.drawContours(image.copy(), contours, -1, color[::-1], thickness)  # BGR


def visualize_predictions(
    image: np.ndarray,
    prediction: Dict[str, Any],
    show_labels: bool = True,
    show_scores: bool = True,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Visualize model predictions on an image.

    Handles different prediction formats:
    - U-Net/AttentionUNet/DynamicRUNext: {mask, cls_probs, predicted_class}
    - Mask R-CNN/Mask2Former: {masks, labels, scores}
    - SegFormer: {semantic_mask}

    Returns RGB image with overlaid predictions.
    """
    if image is None or (isinstance(image, np.ndarray) and image.size == 0):
        return np.zeros((640, 640, 3), dtype=np.uint8)

    vis = image.copy().astype(np.uint8)
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2RGB)

    # Handle single binary mask (U-Net style)
    if "mask" in prediction and "predicted_class" in prediction:
        mask = prediction["mask"]
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()
        cls_id = prediction.get("predicted_class", 1)
        score = prediction.get("cls_probs", None)
        color = CLASS_COLORS.get(cls_id, (255, 255, 0))

        vis = draw_mask(vis, mask, color, alpha)
        vis = draw_contour(vis, mask, color)

        if show_labels:
            label_text = CLASS_NAMES.get(cls_id, f"Class {cls_id}")
            if show_scores and score is not None:
                if isinstance(score, torch.Tensor):
                    score_val = float(score[cls_id - 1]) if score.dim() > 0 else float(score)
                else:
                    score_val = float(score)
                label_text += f" {score_val:.2f}"
            # Find centroid of mask for label placement
            ys, xs = np.where(mask > 0)
            if len(xs) > 0:
                cx, cy = int(xs.mean()), int(ys.mean())
                cv2.putText(vis, label_text, (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Handle per-instance masks (Mask R-CNN / Mask2Former style)
    elif "masks" in prediction:
        masks = prediction["masks"]
        labels = prediction.get("labels", [])
        scores = prediction.get("scores", [])

        if isinstance(masks, torch.Tensor):
            masks = masks.numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy().tolist()
        if isinstance(scores, torch.Tensor):
            scores = scores.numpy().tolist()

        for i, mask in enumerate(masks):
            if mask.sum() == 0:
                continue
            cls_id = int(labels[i]) if i < len(labels) else 1
            score = scores[i] if i < len(scores) else 1.0
            color = CLASS_COLORS.get(cls_id, (255, 255, 0))

            vis = draw_mask(vis, mask, color, alpha)
            vis = draw_contour(vis, mask, color)

            if show_labels:
                label_text = CLASS_NAMES.get(cls_id, f"Class {cls_id}")
                if show_scores:
                    label_text += f" {score:.2f}"
                ys, xs = np.where(mask > 0)
                if len(xs) > 0:
                    cx, cy = int(xs.mean()), int(ys.mean())
                    cv2.putText(vis, label_text, (cx, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Handle semantic mask (SegFormer style)
    elif "semantic_mask" in prediction:
        sem_mask = prediction["semantic_mask"]
        if isinstance(sem_mask, torch.Tensor):
            sem_mask = sem_mask.numpy()

        for cls_id in [1, 2, 3]:
            mask = (sem_mask == cls_id)
            if mask.sum() == 0:
                continue
            color = CLASS_COLORS.get(cls_id, (255, 255, 0))
            vis = draw_mask(vis, mask, color, alpha)
            vis = draw_contour(vis, mask.astype(np.uint8), color)

    return vis


def save_visualization(
    image: np.ndarray,
    prediction: Dict[str, Any],
    output_path: str,
    **kwargs,
):
    """Save visualization to file."""
    vis = visualize_predictions(image, prediction, **kwargs)
    cv2.imwrite(output_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


def plot_comparison(
    images: List[np.ndarray],
    predictions: List[Dict[str, Any]],
    titles: List[str] = None,
    cols: int = 3,
    figsize: tuple = (15, 10),
    save_path: Optional[str] = None,
):
    """Plot multiple prediction visualizations in a grid."""
    n = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if n > 1 else [axes]

    for i, (img, pred) in enumerate(zip(images, predictions)):
        vis = visualize_predictions(img, pred)
        axes[i].imshow(vis)
        if titles:
            axes[i].set_title(titles[i])
        axes[i].axis("off")

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].axis("off")

    # Legend
    legend_patches = [
        mpatches.Patch(color=np.array(c) / 255, label=name)
        for cls_id, (c, name) in {
            1: (CLASS_COLORS[1], "Fiber"),
            2: (CLASS_COLORS[2], "Fragment"),
            3: (CLASS_COLORS[3], "Film"),
        }.items()
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")
    plt.show()
