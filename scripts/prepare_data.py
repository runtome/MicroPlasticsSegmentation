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
    class_names = data_cfg.get("class_names", ["Fiber", "Fragment", "Film"])

    # Step 1: Build splits
    print("=" * 60)
    print("Step 1: Building train/val/test splits + 5-fold CV")
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

    # Step 2: Validate images exist
    print("\n" + "=" * 60)
    print("Step 2: Validating images")
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

    # Step 3: YOLO format conversion
    if not args.no_yolo:
        print("\n" + "=" * 60)
        print("Step 3: Converting to YOLO segmentation format")
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
