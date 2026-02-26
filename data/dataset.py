"""
MicroPlastics COCO Dataset.
Returns per-instance masks + class labels from COCO polygon annotations.
"""
import os
import json
from pathlib import Path
from typing import Optional, Callable, List, Dict, Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class MicroPlasticsDataset(Dataset):
    """
    COCO-format instance segmentation dataset for microplastics.

    Each item returns:
        image: (C, H, W) float32 tensor, normalized
        masks: (N, H, W) binary float32 tensor — one mask per instance
        labels: (N,) int64 tensor — category id (1-indexed, 1=Fiber, 2=Fragment, 3=Film)
        boxes:  (N, 4) float32 tensor — [x1, y1, x2, y2]
        image_id: int
        file_name: str
    """

    def __init__(
        self,
        images_dir: str,
        annotation_path: str,
        file_names: Optional[List[str]] = None,
        transforms: Optional[Callable] = None,
        image_size: int = 640,
    ):
        self.images_dir = Path(images_dir)
        self.transforms = transforms
        self.image_size = image_size

        with open(annotation_path) as f:
            coco = json.load(f)

        self.categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

        # Filter to requested file names if given
        if file_names is not None:
            file_name_set = set(file_names)
            self.images = [img for img in coco["images"] if img["file_name"] in file_name_set]
        else:
            self.images = coco["images"]

        self.image_id_to_meta = {img["id"]: img for img in self.images}

        # Build image_id -> annotations mapping
        self._ann_by_image: Dict[int, List[Dict]] = {}
        for ann in coco["annotations"]:
            iid = ann["image_id"]
            if iid in self.image_id_to_meta:
                self._ann_by_image.setdefault(iid, []).append(ann)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_meta = self.images[idx]
        image_id = img_meta["id"]
        file_name = img_meta["file_name"]

        # Load image
        img_path = self.images_dir / file_name
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        H, W = image.shape[:2]

        # Build per-instance masks
        annotations = self._ann_by_image.get(image_id, [])
        masks = []
        labels = []
        boxes = []

        for ann in annotations:
            mask = np.zeros((H, W), dtype=np.uint8)
            for seg in ann.get("segmentation", []):
                if len(seg) < 6:
                    continue
                pts = np.array(seg, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [pts], 1)
            masks.append(mask)
            labels.append(ann["category_id"])

            # Bounding box from COCO [x, y, w, h] — clip to image bounds
            x, y, w, h = ann["bbox"]
            x1 = max(0.0, x)
            y1 = max(0.0, y)
            x2 = min(float(W), x + w)
            y2 = min(float(H), y + h)
            boxes.append([x1, y1, x2, y2])

        # Apply transforms (albumentations format)
        if self.transforms is not None:
            if masks:
                result = self.transforms(
                    image=image,
                    masks=masks,
                    bboxes=[[b[0], b[1], b[2], b[3]] for b in boxes],
                    category_ids=labels,
                )
                image = result["image"]
                masks = result["masks"]
                boxes = result["bboxes"]
                labels = result["category_ids"]
            else:
                result = self.transforms(image=image, masks=[], bboxes=[], category_ids=[])
                image = result["image"]

        # Convert to tensors
        # image: already normalized by albumentations ToTensorV2 or we do it here
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        if masks:
            masks_tensor = torch.stack([
                torch.from_numpy(np.array(m, dtype=np.float32)) for m in masks
            ])  # (N, H, W)
            labels_tensor = torch.tensor(labels, dtype=torch.long)
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        else:
            h, w = self.image_size, self.image_size
            masks_tensor = torch.zeros((0, h, w), dtype=torch.float32)
            labels_tensor = torch.zeros(0, dtype=torch.long)
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)

        return {
            "image": image,
            "masks": masks_tensor,
            "labels": labels_tensor,
            "boxes": boxes_tensor,
            "image_id": image_id,
            "file_name": file_name,
        }


def collate_fn(batch):
    """Custom collate for variable-length instance lists."""
    images = torch.stack([item["image"] for item in batch])
    masks = [item["masks"] for item in batch]
    labels = [item["labels"] for item in batch]
    boxes = [item["boxes"] for item in batch]
    image_ids = [item["image_id"] for item in batch]
    file_names = [item["file_name"] for item in batch]

    return {
        "image": images,
        "masks": masks,
        "labels": labels,
        "boxes": boxes,
        "image_ids": image_ids,
        "file_names": file_names,
    }
