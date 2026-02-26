"""
COCO polygon annotations → binary instance masks.
Utility functions used by dataset.py.
"""
import numpy as np
import cv2


def polygon_to_mask(segmentation: list, height: int, width: int) -> np.ndarray:
    """
    Convert COCO segmentation polygons to a single binary mask.

    Args:
        segmentation: list of polygon lists [[x1,y1,x2,y2,...], ...]
        height: image height
        width: image width

    Returns:
        Binary mask (H, W) uint8 with 1 inside the polygon(s).
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for seg in segmentation:
        if len(seg) < 6:
            continue
        pts = np.array(seg, dtype=np.float32).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def annotations_to_masks(annotations: list, height: int, width: int):
    """
    Convert a list of COCO annotations for one image to per-instance masks.

    Returns:
        masks: list of (H, W) uint8 arrays
        labels: list of int category_ids
        boxes: list of [x1, y1, x2, y2]
    """
    masks, labels, boxes = [], [], []
    for ann in annotations:
        mask = polygon_to_mask(ann.get("segmentation", []), height, width)
        masks.append(mask)
        labels.append(ann["category_id"])
        x, y, w, h = ann["bbox"]
        boxes.append([x, y, x + w, y + h])
    return masks, labels, boxes
