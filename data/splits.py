"""
Build train/val/test splits and 5-fold CV splits.
Groups Roboflow augmented variants with their base image to prevent data leakage.
"""
import json
import re
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold


def _strip_rf_suffix(filename: str) -> str:
    """Strip Roboflow suffix: 'base_jpg.rf.HASH.jpg' -> 'base'."""
    name = Path(filename).stem  # remove extension
    # Match pattern: anything followed by .rf. and a hex hash
    match = re.match(r"^(.+?)_jpg\.rf\.[0-9a-f]+$", name)
    if match:
        return match.group(1)
    # Also handle direct _jpg suffix (non-augmented Roboflow images)
    match2 = re.match(r"^(.+?)_jpg$", name)
    if match2:
        return match2.group(1)
    return name  # original image, no suffix


def _get_dominant_class(image_id: int, annotations: list) -> int:
    """Return the most frequent class label among annotations for this image."""
    ann_for_img = [a for a in annotations if a["image_id"] == image_id]
    if not ann_for_img:
        return 0  # unannotated
    from collections import Counter
    counts = Counter(a["category_id"] for a in ann_for_img)
    return counts.most_common(1)[0][0]


def build_splits(
    annotation_path: str,
    output_path: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    num_folds: int = 5,
    seed: int = 42,
) -> dict:
    """
    Parse COCO annotation file, group augmented variants by base image,
    perform stratified split, and optionally build 5-fold CV indices.

    Returns a dict with keys: train, val, test, folds (list of {train, val}).
    Each value is a list of image file names.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    with open(annotation_path) as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]

    # Build image_id -> filename mapping
    id_to_file = {img["id"]: img["file_name"] for img in images}

    # Group images by base name (strip RF suffix)
    base_to_images = defaultdict(list)
    for img in images:
        base = _strip_rf_suffix(img["file_name"])
        base_to_images[base].append(img)

    # Assign dominant class to each base group (for stratification)
    base_names = sorted(base_to_images.keys())

    ann_by_image = defaultdict(list)
    for ann in annotations:
        ann_by_image[ann["image_id"]].append(ann)

    base_labels = []
    for base in base_names:
        imgs_in_group = base_to_images[base]
        # Aggregate annotations for all images in group
        all_anns = []
        for img in imgs_in_group:
            all_anns.extend(ann_by_image[img["id"]])
        if all_anns:
            from collections import Counter
            counts = Counter(a["category_id"] for a in all_anns)
            label = counts.most_common(1)[0][0]
        else:
            label = 0
        base_labels.append(label)

    base_names = np.array(base_names)
    base_labels = np.array(base_labels)

    rng = random.Random(seed)
    np.random.seed(seed)

    # Stratified split at base-group level
    n = len(base_names)
    indices = np.arange(n)

    # Sort by label for stratified sampling
    sorted_idx = np.argsort(base_labels, kind="stable")
    sorted_names = base_names[sorted_idx]
    sorted_labels = base_labels[sorted_idx]

    train_names, val_names, test_names = [], [], []
    label_groups = defaultdict(list)
    for i, lbl in enumerate(sorted_labels):
        label_groups[lbl].append(sorted_names[i])

    for lbl, names in label_groups.items():
        names = list(names)
        rng.shuffle(names)
        n_group = len(names)
        n_test = max(1, round(n_group * test_ratio))
        n_val = max(1, round(n_group * val_ratio))
        n_train = n_group - n_test - n_val

        train_names.extend(names[:n_train])
        val_names.extend(names[n_train:n_train + n_val])
        test_names.extend(names[n_train + n_val:])

    def expand_to_files(base_list):
        files = []
        for base in base_list:
            for img in base_to_images[base]:
                files.append(img["file_name"])
        return files

    train_files = expand_to_files(train_names)
    val_files = expand_to_files(val_names)
    test_files = expand_to_files(test_names)

    # 5-fold CV on train set (base-level stratified)
    train_bases = np.array(train_names)
    train_base_labels = np.array([
        base_labels[np.where(base_names == b)[0][0]] for b in train_bases
    ])

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    folds = []
    for fold_train_idx, fold_val_idx in skf.split(train_bases, train_base_labels):
        fold_train_files = expand_to_files(train_bases[fold_train_idx].tolist())
        fold_val_files = expand_to_files(train_bases[fold_val_idx].tolist())
        folds.append({"train": fold_train_files, "val": fold_val_files})

    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files,
        "folds": folds,
        "stats": {
            "total_images": len(images),
            "annotated": len([img for img in images if img["id"] in ann_by_image]),
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files),
            "base_groups": len(base_names),
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Splits saved to {output_path}")
    print(f"  Train: {len(train_files)} images ({len(train_names)} groups)")
    print(f"  Val:   {len(val_files)} images ({len(val_names)} groups)")
    print(f"  Test:  {len(test_files)} images ({len(test_names)} groups)")
    print(f"  5-fold CV folds built on train set")

    return splits
