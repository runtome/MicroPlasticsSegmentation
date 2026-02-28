#!/usr/bin/env python3
"""
Visualize each augmentation transform per class.

For every combination of:
    augmentation  × class (Fiber / Fragment / Film)

Saves a side-by-side JPG:  Original  |  Augmented
Filename: {AugmentName}_{ClassName}.jpg

Output directory: paper_figures/transforms/

Usage:
    python visualization/visualization_transform.py
    python visualization/visualization_transform.py --images-dir /path/to/images
                                                    --anno     /path/to/annotation.json
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import albumentations as A

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent.parent
OUT_DIR = ROOT / "paper_figures" / "transforms"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Dataset constants ─────────────────────────────────────────────────────────
CLASSES = {1: "Fiber", 2: "Fragment", 3: "Film"}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMAGE_SIZE = 640

# ── Individual augmentations (p=1.0 → always applied) ────────────────────────
def build_augmentations() -> dict:
    """Return ordered dict of name → albumentations transform."""
    augs = {}

    augs["HorizontalFlip"] = A.HorizontalFlip(p=1.0)
    augs["VerticalFlip"]   = A.VerticalFlip(p=1.0)
    augs["RandomRotate90"] = A.RandomRotate90(p=1.0)

    augs["ColorJitter"] = A.ColorJitter(
        brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15, p=1.0
    )

    augs["GaussNoise"] = A.GaussNoise(p=1.0)

    augs["LongestMaxSize+Pad"] = A.Compose([
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=0),
    ])

    return augs


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_coco(anno_path: Path) -> dict:
    with open(anno_path) as f:
        return json.load(f)


def find_samples(coco: dict, images_dir: Path) -> dict:
    """
    Return {class_id: image_path} for one representative image per class.
    Prefers images that ONLY contain that class for clearest visualization.
    Falls back to any image containing the class.
    """
    img_to_cats: dict = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        img_to_cats.setdefault(img_id, set()).add(ann["category_id"])

    id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}

    samples = {}
    for cls_id in CLASSES:
        # prefer exclusive images first, then any
        for exclusive in [True, False]:
            for img_id, cats in img_to_cats.items():
                if cls_id not in cats:
                    continue
                if exclusive and len(cats) > 1:
                    continue
                path = images_dir / id_to_file[img_id]
                if path.exists():
                    samples[cls_id] = path
                    break
            if cls_id in samples:
                break

    return samples


def load_and_resize(img_path: Path, size: int = IMAGE_SIZE) -> np.ndarray:
    """Load image as RGB uint8, resize to square."""
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        raise FileNotFoundError(f"Cannot load image: {img_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    return rgb


def apply_aug(image_rgb: np.ndarray, aug) -> np.ndarray:
    """Apply a single albumentations transform to an RGB uint8 image."""
    result = aug(image=image_rgb)
    return result["image"].astype(np.uint8)


def save_comparison(
    original: np.ndarray,
    augmented: np.ndarray,
    aug_name: str,
    cls_name: str,
    out_dir: Path,
) -> Path:
    """
    Save side-by-side comparison:
      Left  — Original
      Right — Augmented (with method name highlighted)
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.patch.set_facecolor("white")

    # ── Left: original ────────────────────────────────────────────────────────
    axes[0].imshow(original)
    axes[0].set_title("Original", fontsize=15, fontweight="bold",
                       color="#1a1a1a", pad=12)
    axes[0].axis("off")
    # thin border
    for spine in axes[0].spines.values():
        spine.set_edgecolor("#AAAAAA")
        spine.set_linewidth(1.2)

    # ── Right: augmented ──────────────────────────────────────────────────────
    axes[1].imshow(augmented)
    axes[1].set_title(aug_name, fontsize=15, fontweight="bold",
                       color="#1565C0", pad=12)
    axes[1].axis("off")
    for spine in axes[1].spines.values():
        spine.set_edgecolor("#1565C0")
        spine.set_linewidth(2.0)

    # ── Main title ────────────────────────────────────────────────────────────
    fig.suptitle(
        f"Augmentation: {aug_name}    |    Class: {cls_name}",
        fontsize=16, fontweight="bold", color="#1a1a1a", y=1.02,
    )

    # ── Class badge (top-left of augmented panel) ─────────────────────────────
    badge_colors = {"Fiber": "#1B5E20", "Fragment": "#0D47A1", "Film": "#B71C1C"}
    badge_col = badge_colors.get(cls_name, "#37474F")
    axes[1].text(
        0.02, 0.97, cls_name,
        transform=axes[1].transAxes,
        fontsize=11, fontweight="bold", color="white",
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=badge_col, alpha=0.85),
    )

    plt.tight_layout(pad=1.5)

    out_path = out_dir / f"{aug_name}_{cls_name}.jpg"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight",
                facecolor="white", pil_kwargs={"quality": 95})
    plt.close()
    return out_path


def save_grid_summary(
    samples: dict,
    augmentations: dict,
    out_dir: Path,
):
    """
    Save one combined grid overview:
        rows = augmentations,  cols = classes
    Each cell shows the augmented image.
    """
    cls_ids  = list(CLASSES.keys())
    aug_names = list(augmentations.keys())
    n_rows = len(aug_names)
    n_cols = len(cls_ids) + 1  # +1 for original column

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 3.5 * n_rows))
    fig.patch.set_facecolor("white")

    badge_colors = {"Fiber": "#1B5E20", "Fragment": "#0D47A1", "Film": "#B71C1C"}

    # Pre-load originals
    originals = {}
    for cls_id in cls_ids:
        if cls_id in samples:
            originals[cls_id] = load_and_resize(samples[cls_id])

    for row, (aug_name, aug) in enumerate(augmentations.items()):
        for col_idx, cls_id in enumerate(cls_ids):
            ax_orig = axes[row, 0] if col_idx == 0 else None
            ax      = axes[row, col_idx + 1]

            if cls_id not in originals:
                ax.axis("off")
                continue

            orig = originals[cls_id]
            aug_img = apply_aug(orig.copy(), aug)

            # Original column (only first aug row for cleanliness)
            if row == 0 and col_idx == 0:
                axes[row, 0].imshow(orig)
                axes[row, 0].set_title("Original\n(Fiber)", fontsize=9,
                                        fontweight="bold")
                axes[row, 0].axis("off")
            elif col_idx == 0:
                # show original for this class
                axes[row, 0].imshow(orig)
                axes[row, 0].axis("off")

            ax.imshow(aug_img)
            ax.axis("off")

            cls_name = CLASSES[cls_id]
            ax.text(0.02, 0.97, cls_name, transform=ax.transAxes,
                    fontsize=8, fontweight="bold", color="white", va="top",
                    bbox=dict(boxstyle="round,pad=0.25",
                              facecolor=badge_colors[cls_name], alpha=0.85))

        # Row label (augmentation name)
        axes[row, 0].set_ylabel(aug_name, fontsize=10, fontweight="bold",
                                 rotation=0, labelpad=80, va="center",
                                 color="#1565C0")

    # Column headers
    col_headers = ["Original"] + [CLASSES[c] for c in cls_ids]
    for col, header in enumerate(col_headers):
        axes[0, col].set_title(header, fontsize=11, fontweight="bold", pad=8)

    fig.suptitle("Augmentation Overview — All Methods × All Classes",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout(pad=0.8)

    out_path = out_dir / "augmentation_overview.jpg"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight",
                facecolor="white", pil_kwargs={"quality": 95})
    plt.close()
    print(f"  Saved grid: {out_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize augmentation transforms per class")
    parser.add_argument("--anno",       default=None, help="Path to annotation.json")
    parser.add_argument("--images-dir", default=None, help="Path to images directory")
    parser.add_argument("--out-dir",    default=str(OUT_DIR), help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Resolve paths ─────────────────────────────────────────────────────────
    if args.anno:
        anno_path  = Path(args.anno)
        images_dir = Path(args.images_dir) if args.images_dir else anno_path.parent / "images"
    else:
        # Try local repo path first, then Kaggle
        candidates = [
            (ROOT / "annotation.json",                           ROOT / "images"),
            (Path("/kaggle/input/microplasticssegmentation/annotation.json"),
             Path("/kaggle/input/microplasticssegmentation/images/")),
        ]
        anno_path, images_dir = None, None
        for ap, ip in candidates:
            if ap.exists():
                anno_path, images_dir = ap, ip
                break

    if anno_path is None or not anno_path.exists():
        print("annotation.json not found. Pass --anno /path/to/annotation.json")
        sys.exit(1)

    print(f"Annotations : {anno_path}")
    print(f"Images dir  : {images_dir}")
    print(f"Output dir  : {out_dir}\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    coco    = load_coco(anno_path)
    samples = find_samples(coco, images_dir)

    if not samples:
        print("No sample images found. Check images_dir path.")
        sys.exit(1)

    for cls_id, path in samples.items():
        print(f"  Class {CLASSES[cls_id]:10s} : {path.name}")

    augmentations = build_augmentations()
    total = len(augmentations) * len(samples)
    print(f"\nGenerating {total} comparison images "
          f"({len(augmentations)} augmentations × {len(samples)} classes)...\n")

    # ── Per-image comparisons ─────────────────────────────────────────────────
    count = 0
    for cls_id, img_path in samples.items():
        cls_name = CLASSES[cls_id]
        print(f"Class: {cls_name}  ({img_path.name})")

        try:
            original = load_and_resize(img_path)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        for aug_name, aug in augmentations.items():
            try:
                augmented = apply_aug(original.copy(), aug)
                out_path  = save_comparison(original, augmented,
                                            aug_name, cls_name, out_dir)
                print(f"  [OK] {out_path.name}")
                count += 1
            except Exception as e:
                print(f"  [FAIL] {aug_name}: {e}")

        print()

    # ── Grid overview ─────────────────────────────────────────────────────────
    print("Generating grid overview...")
    try:
        save_grid_summary(samples, augmentations, out_dir)
    except Exception as e:
        print(f"  Grid failed: {e}")

    print(f"\nDone. {count} images saved to: {out_dir}")


if __name__ == "__main__":
    main()
