"""
Run inference on images and save predictions in COCO annotation format.

The output JSON matches the structure of annotation.json:
  {
    "images": [...],
    "categories": [...],
    "annotations": [...]
  }

Each detected connected component in the predicted mask becomes one annotation entry
with polygon segmentation, bbox, area, and category_id.

Usage:
    # Named splits from splits.json (one or more):
    python scripts/predict_coco.py --model unet --split test
    python scripts/predict_coco.py --model unet --split train val test

    # File / folder paths (one or more):
    python scripts/predict_coco.py --model unet --split images/
    python scripts/predict_coco.py --model unet --split images/ extra/img.jpg

    # Mixed named splits and paths:
    python scripts/predict_coco.py --model unet --split test images/unlabelled/

    # Single image or directory via --input (unchanged):
    python scripts/predict_coco.py --model unet --input images/

    # Explicit checkpoint — path, directory, or 'best' keyword (default):
    python scripts/predict_coco.py --model unet --checkpoint outputs/checkpoints/unet/best.pth --split test

    # Specify output file:
    python scripts/predict_coco.py --model unet --split test \
        --output outputs/predictions/test_predictions.json
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


CATEGORIES = [
    {"id": 1, "name": "Fiber"},
    {"id": 2, "name": "Fragment"},
    {"id": 3, "name": "Film"},
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ---------------------------------------------------------------------------
# Mask utilities
# ---------------------------------------------------------------------------

def _connected_components(binary_mask: np.ndarray):
    """
    Split a binary mask into per-instance masks using connected components.
    Returns list of (instance_mask, stats) where stats = [x, y, w, h, area].
    Ignores background label 0.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask.astype(np.uint8), connectivity=8
    )
    instances = []
    for label_id in range(1, num_labels):  # skip background (0)
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < 10:  # skip tiny noise blobs
            continue
        instance_mask = (labels == label_id).astype(np.uint8)
        instances.append((instance_mask, stats[label_id]))
    return instances


def _mask_to_polygon(instance_mask: np.ndarray):
    """
    Convert a binary instance mask to a COCO polygon segmentation list.
    Returns list of flat [x1, y1, x2, y2, ...] arrays (one per contour).
    Returns empty list if no valid contour found.
    """
    contours, _ = cv2.findContours(
        instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons = []
    for contour in contours:
        if contour.shape[0] < 3:
            continue
        poly = contour.flatten().tolist()
        if len(poly) >= 6:  # need at least 3 points
            polygons.append(poly)
    return polygons


def _scale_polygon(polygons, sx: float, sy: float):
    """Scale polygon coordinates by (sx, sy)."""
    scaled = []
    for poly in polygons:
        pts = np.array(poly, dtype=np.float64).reshape(-1, 2)
        pts[:, 0] *= sx
        pts[:, 1] *= sy
        scaled.append(pts.flatten().tolist())
    return scaled


def _scale_bbox(x, y, w, h, sx: float, sy: float):
    """Scale COCO bbox [x, y, w, h] by (sx, sy). Returns rounded ints."""
    return [
        round(x * sx, 2),
        round(y * sy, 2),
        round(w * sx, 2),
        round(h * sy, 2),
    ]


def _polygon_area(polygons):
    """Compute area from polygon using the shoelace formula."""
    total = 0.0
    for poly in polygons:
        pts = np.array(poly, dtype=np.float64).reshape(-1, 2)
        n = len(pts)
        if n < 3:
            continue
        x, y = pts[:, 0], pts[:, 1]
        total += 0.5 * abs(
            np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
        )
    return round(total, 4)


# ---------------------------------------------------------------------------
# Core: build COCO annotations from predictions
# ---------------------------------------------------------------------------

def _build_annotations_for_image(
    pred: dict,
    image_id: int,
    orig_h: int,
    orig_w: int,
    model_size: int,
    ann_id_start: int,
) -> list:
    """
    Convert a model prediction dict to a list of COCO annotation dicts.

    pred keys (U-Net style):
        "mask"            : (H, W) bool/uint8 tensor — binary segmentation mask
        "predicted_class" : int — 1-indexed class id for the whole image
        (optional) "masks"  : (N, H, W) — per-instance masks (e.g. Mask R-CNN)
        (optional) "labels" : (N,)  — per-instance 1-indexed class ids
        (optional) "scores" : (N,)  — per-instance confidence scores

    Returns list of annotation dicts ready for COCO JSON.
    """

    annotations = []
    ann_id = ann_id_start

    # Scale factors: from model space → original image space.
    # The data pipeline uses LongestMaxSize(model_size) + PadIfNeeded(model_size),
    # which preserves aspect ratio and pads to square.  Both axes share the
    # same scale factor (the inverse of the resize ratio).
    resize_scale = model_size / max(orig_h, orig_w)
    sx = 1.0 / resize_scale
    sy = 1.0 / resize_scale

    # ---- compute resized shape BEFORE padding ----
    resized_h = int(round(orig_h * resize_scale))
    resized_w = int(round(orig_w * resize_scale))

    # ---- compute padding applied in model space ----
    pad_x = (model_size - resized_w) / 2.0
    pad_y = (model_size - resized_h) / 2.0

    # ---- helpers ----
    def remove_padding_from_polygon(polygons):
        fixed = []
        for poly in polygons:
            new_poly = []
            for i in range(0, len(poly), 2):
                x = poly[i]   - pad_x
                y = poly[i+1] - pad_y
                new_poly.extend([x, y])
            fixed.append(new_poly)
        return fixed

    def remove_padding_from_bbox(x, y, w, h):
        return x - pad_x, y - pad_y, w, h

    # ────────────────────────────────────────────────
    # Case 1: per-instance masks (Mask R-CNN style)
    # ────────────────────────────────────────────────
    if "masks" in pred and len(pred["masks"]) > 0:
        masks  = pred["masks"]   # (N, H, W) numpy or tensor
        labels = pred.get("labels", [])
        scores = pred.get("scores", [])

        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        for i, inst_mask in enumerate(masks):
            inst_mask = (inst_mask > 0).astype(np.uint8)

            polygons = _mask_to_polygon(inst_mask)
            if not polygons:
                continue

            polygons = remove_padding_from_polygon(polygons)
            polygons = _scale_polygon(polygons, sx, sy)
            area = _polygon_area(polygons)

            # bbox from mask (in model space → scale)
            rows = np.any(inst_mask, axis=1)
            cols = np.any(inst_mask, axis=0)
            if not rows.any():
                continue

            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]

            x, y, w, h = remove_padding_from_bbox(
                cmin, rmin, cmax - cmin + 1, rmax - rmin + 1
            )
            bbox = _scale_bbox(x, y, w, h, sx, sy)

            cat_id = int(labels[i]) if i < len(labels) else 1
            score  = float(scores[i]) if i < len(scores) else None

            ann = {
                "id": ann_id,
                "image_id": image_id,
                "category_id": cat_id,
                "area": area,
                "iscrowd": 0,
                "segmentation": polygons,
                "bbox": bbox,
            }
            if score is not None:
                ann["score"] = round(score, 4)

            annotations.append(ann)
            ann_id += 1

    # ────────────────────────────────────────────────
    # Case 2: single binary mask (U-Net style)
    # ────────────────────────────────────────────────
    elif "mask" in pred:
        mask = pred["mask"]
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        mask = (mask > 0).astype(np.uint8)

        cat_id = int(pred.get("predicted_class", 1))

        instances = _connected_components(mask)
        for inst_mask, stats in instances:
            polygons = _mask_to_polygon(inst_mask)
            if not polygons:
                continue

            polygons = remove_padding_from_polygon(polygons)
            polygons = _scale_polygon(polygons, sx, sy)
            area = _polygon_area(polygons)

            x = int(stats[cv2.CC_STAT_LEFT])
            y = int(stats[cv2.CC_STAT_TOP])
            w = int(stats[cv2.CC_STAT_WIDTH])
            h = int(stats[cv2.CC_STAT_HEIGHT])

            x, y, w, h = remove_padding_from_bbox(x, y, w, h)
            bbox = _scale_bbox(x, y, w, h, sx, sy)

            ann = {
                "id": ann_id,
                "image_id": image_id,
                "category_id": cat_id,
                "area": area,
                "iscrowd": 0,
                "segmentation": polygons,
                "bbox": bbox,
            }

            annotations.append(ann)
            ann_id += 1

    return annotations


# ---------------------------------------------------------------------------
# High-level runners
# ---------------------------------------------------------------------------

def _collect_image_paths(input_path: Path):
    if input_path.is_dir():
        return sorted(
            p for p in input_path.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
    elif input_path.is_file():
        return [input_path]
    else:
        raise FileNotFoundError(f"Input not found: {input_path}")


def _get_image_dims(path: Path):
    """Return (height, width) of an image file without loading all channels."""
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return img.shape[0], img.shape[1]


def run_predict_coco(predictor, image_paths, model_size: int, threshold: float):
    """
    Run inference on a list of image paths.
    Returns COCO-format dict: {images, annotations, categories}.
    """
    coco_images = []
    coco_annotations = []
    ann_id = 1

    for img_id, img_path in enumerate(image_paths, start=1):
        print(f"  [{img_id}/{len(image_paths)}] {img_path.name}", end="", flush=True)

        orig_h, orig_w = _get_image_dims(img_path)

        pred = predictor.predict_single(str(img_path), threshold=threshold)

        anns = _build_annotations_for_image(
            pred,
            image_id=img_id,
            orig_h=orig_h,
            orig_w=orig_w,
            model_size=model_size,
            ann_id_start=ann_id,
        )
        ann_id += len(anns)

        coco_images.append({
            "id": img_id,
            "file_name": img_path.name,
            "height": orig_h,
            "width": orig_w,
        })
        coco_annotations.extend(anns)
        print(f"  → {len(anns)} instance(s)")

    return {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": CATEGORIES,
    }


_NAMED_SPLITS = {"train", "val", "test"}


def _resolve_split_paths(split_values: list, config: dict) -> tuple:
    """
    Resolve a list of split tokens into (image_paths, tag).

    Each token is either:
      - A named split keyword ("train", "val", "test") → resolved via splits.json
      - A directory path  → all images inside it
      - A file path       → that single file

    Returns:
        image_paths : list of Path (deduplicated, order preserved)
        tag         : str label used for the default output filename
    """
    data_cfg = config["data"]
    splits_cache = {}  # lazily load splits.json once

    def _load_splits():
        if not splits_cache:
            with open(data_cfg["splits_file"]) as f:
                splits_cache.update(json.load(f))
        return splits_cache

    image_paths = []
    seen = set()
    named_used = []
    path_used = []

    for token in split_values:
        if token in _NAMED_SPLITS:
            splits = _load_splits()
            file_names = splits.get(token, [])
            if not file_names:
                raise RuntimeError(
                    f"No images found for split '{token}' in {data_cfg['splits_file']}"
                )
            images_dir = Path(data_cfg["images_dir"])
            for fn in file_names:
                p = images_dir / fn
                if p not in seen:
                    image_paths.append(p)
                    seen.add(p)
            named_used.append(token)
            print(f"  Named split '{token}': {len(file_names)} images")
        else:
            p = Path(token)
            if p.is_dir():
                found = sorted(
                    x for x in p.iterdir() if x.suffix.lower() in IMAGE_EXTENSIONS
                )
                for x in found:
                    if x not in seen:
                        image_paths.append(x)
                        seen.add(x)
                path_used.append(str(p))
                print(f"  Folder '{p}': {len(found)} images")
            elif p.is_file():
                if p not in seen:
                    image_paths.append(p)
                    seen.add(p)
                path_used.append(str(p))
                print(f"  File '{p}'")
            else:
                raise FileNotFoundError(
                    f"'{token}' is not a valid split name (train/val/test) "
                    f"nor an existing file or directory."
                )

    # Build a short tag for the default output filename
    if named_used and not path_used:
        tag = "+".join(named_used)
    elif path_used and not named_used:
        tag = "custom"
    else:
        tag = "+".join(named_used) + "+custom"

    return image_paths, tag


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run inference and save COCO-format annotation JSON"
    )
    parser.add_argument("--model", required=True,
                        help="Model name (unet, attention_unet, mask_rcnn, etc.)")
    parser.add_argument("--checkpoint", default="best",
                        help="Checkpoint path, directory, or 'best' (default) to "
                             "auto-select the best .pth from the model's checkpoint dir")
    parser.add_argument("--input", default=None,
                        help="Input image path or directory (required when --split is not used)")
    parser.add_argument("--output", default=None,
                        help="Output JSON file path (default: outputs/predictions/<model>_coco.json)")
    parser.add_argument("--config", default=None,
                        help="Config YAML (default: configs/{model}.yaml)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Mask threshold (default 0.5)")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--split", nargs="+", default=None, metavar="SPLIT_OR_PATH",
        help=(
            "One or more named splits (train / val / test) and/or file/folder paths. "
            "Named splits are resolved via splits.json; paths are globbed for images. "
            "Examples:  --split test  |  --split train val  |  --split images/  "
            "|  --split test images/extra/"
        ),
    )
    args = parser.parse_args()

    if not args.split and not args.input:
        parser.error("--input is required when --split is not specified")

    # ── Config ──────────────────────────────────────────────────────────────
    config_path = args.config or f"configs/{args.model}.yaml"
    from scripts.train import load_config, get_model
    config = load_config(config_path)
    device = args.device or config.get("training", {}).get("device", "cuda")
    model_size = config.get("data", {}).get("image_size", 640)

    # ── Checkpoint ──────────────────────────────────────────────────────────
    from scripts.evaluate import resolve_checkpoint
    model_name = config.get("model", {}).get("name", args.model)
    ckpt_path = resolve_checkpoint(args.checkpoint, model_name, config)
    print(f"Loading checkpoint: {ckpt_path}")

    # ── Model ───────────────────────────────────────────────────────────────
    model = get_model(config)
    state = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)

    from inference.predictor import Predictor
    predictor = Predictor(model, image_size=model_size, device=device)

    # ── Run inference ────────────────────────────────────────────────────────
    if args.split:
        print(f"Resolving --split: {args.split}")
        image_paths, tag = _resolve_split_paths(args.split, config)
        if not image_paths:
            print("No images found — nothing to do.")
            sys.exit(0)
        print(f"Total: {len(image_paths)} image(s)")
        coco = run_predict_coco(predictor, image_paths, model_size, args.threshold)
    else:
        input_path = Path(args.input)
        image_paths = _collect_image_paths(input_path)
        tag = "images"
        print(f"Running inference on {len(image_paths)} image(s) from {input_path}")
        coco = run_predict_coco(predictor, image_paths, model_size, args.threshold)

    # ── Save JSON ────────────────────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path("outputs/predictions") / f"{args.model}_{tag}_coco.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"\nSaved COCO annotation JSON → {out_path}")
    print(f"  images     : {len(coco['images'])}")
    print(f"  annotations: {len(coco['annotations'])}")
    print(f"  categories : {[c['name'] for c in coco['categories']]}")


if __name__ == "__main__":
    main()
