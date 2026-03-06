"""
Prepare data splits and YOLO format conversion.

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --config configs/base.yaml
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Prepare data splits and convert to YOLO format")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--no-yolo", action="store_true", help="Skip YOLO conversion")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]

    annotation_path = data_cfg["annotation"]
    images_dir = data_cfg["images_dir"]
    splits_file = data_cfg["splits_file"]
    image_size = data_cfg.get("image_size", 640)
    class_names = data_cfg.get("class_names", ["Fiber", "Fragment"])

    # Step 1: Dataset statistics
    print("=" * 60)
    print("Step 1: Dataset statistics")
    print("=" * 60)
    with open(annotation_path) as f:
        coco = json.load(f)

    num_images = len(coco["images"])
    num_annotations = len(coco["annotations"])
    num_categories = len(coco["categories"])

    print(f"Total images      : {num_images}")
    print(f"Total annotations : {num_annotations}")
    print(f"Total categories  : {num_categories}")

    cat_id_to_name = {cat["id"]: cat["name"] for cat in coco["categories"]}
    cat_counter = Counter(ann["category_id"] for ann in coco["annotations"])

    print("\nAnnotation count per class:")
    print(f'  {"class_name":<12} {"count":>8} {"percentage":>10}')
    print(f'  {"-"*12} {"-"*8} {"-"*10}')
    for cat_id, count in sorted(cat_counter.items()):
        pct = round(count / num_annotations * 100, 2)
        print(f"  {cat_id_to_name[cat_id]:<12} {count:>8} {pct:>9}%")

    image_classes = defaultdict(set)
    for ann in coco["annotations"]:
        image_classes[ann["image_id"]].add(ann["category_id"])

    images_per_class = Counter()
    for cats in image_classes.values():
        for cat_id in cats:
            images_per_class[cat_id] += 1

    print(f"\nImages containing each class (out of {num_images} total):")
    print(f'  {"class_name":<12} {"images":>8} {"pct_of_imgs":>12}')
    print(f'  {"-"*12} {"-"*8} {"-"*12}')
    for cat_id in sorted(cat_id_to_name):
        n = images_per_class.get(cat_id, 0)
        pct = round(n / num_images * 100, 2)
        print(f"  {cat_id_to_name[cat_id]:<12} {n:>8} {pct:>11}%")

    annotated_ids = {ann["image_id"] for ann in coco["annotations"]}
    all_ids = {img["id"] for img in coco["images"]}
    unannotated = all_ids - annotated_ids

    print(f"\nImages with at least one annotation : {len(annotated_ids)}")
    print(f"Images with NO annotation           : {len(unannotated)}")

    # Step 2: Build splits
    print("\n" + "=" * 60)
    print("Step 2: Building train/val/test splits + 5-fold CV")
    print("=" * 60)
    from data.splits import build_splits
    splits = build_splits(
        annotation_path=annotation_path,
        output_path=splits_file,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        num_folds=5,
        seed=42,
    )

    print("\nSplit statistics:")
    stats = splits["stats"]
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Step 3: Validate images exist
    print("\n" + "=" * 60)
    print("Step 3: Validating images")
    print("=" * 60)
    missing = []
    for split_name in ["train", "val", "test"]:
        for fname in splits[split_name]:
            img_path = Path(images_dir) / fname
            if not img_path.exists():
                missing.append(str(img_path))

    if missing:
        print(f"WARNING: {len(missing)} images not found!")
        for p in missing[:10]:
            print(f"  {p}")
    else:
        total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        print(f"All {total} images validated OK.")

    # Step 4: YOLO format conversion
    if not args.no_yolo:
        print("\n" + "=" * 60)
        print("Step 4: Converting to YOLO segmentation format")
        print("=" * 60)
        from data.converters.to_yolo import coco_to_yolo_seg
        yolo_output_dir = "data_splits/yolo"
        coco_to_yolo_seg(
            annotation_path=annotation_path,
            images_dir=images_dir,
            output_dir=yolo_output_dir,
            splits=splits,
            class_names=class_names,
        )
    else:
        print("\nSkipping YOLO conversion (--no-yolo)")

    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print(f"Splits saved to: {splits_file}")
    if not args.no_yolo:
        print("YOLO dataset: data_splits/yolo/dataset.yaml")
    print("=" * 60)


if __name__ == "__main__":
    main()
