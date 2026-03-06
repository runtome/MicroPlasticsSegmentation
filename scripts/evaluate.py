"""
Evaluate a trained model checkpoint on the test set.

Usage:
    python scripts/evaluate.py --config configs/unet.yaml --checkpoint outputs/checkpoints/unet/unet_best.pth
    python scripts/evaluate.py --config configs/unet.yaml --checkpoint path/to/ckpt.pth --split test
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import yaml


def load_config(config_path: str) -> dict:
    from scripts.train import load_config as _load
    return _load(config_path)


def resolve_checkpoint(ckpt_arg: str, model_name: str, config: dict) -> str:
    """
    Resolve checkpoint path flexibly:
      - Exact .pth file path  → use as-is
      - Directory path        → find best .pth inside (highest val_miou in filename)
      - 'best'                → search default checkpoint dir for best .pth
      - Bare filename         → search default checkpoint dir
    Prints available checkpoints if nothing is found.
    """
    p = Path(ckpt_arg)

    # 1. Exact file exists → done
    if p.is_file():
        return str(p)

    # 2. Directory given → find best .pth inside it
    if p.is_dir():
        return _best_in_dir(p, ckpt_arg)

    # 3. "best" keyword → search default checkpoint dir
    default_dir = Path(
        config.get("output", {}).get("checkpoint_dir",
                                     f"outputs/checkpoints/{model_name}/")
    )
    if ckpt_arg.lower() == "best":
        if default_dir.is_dir():
            return _best_in_dir(default_dir, str(default_dir))
        # Also search runs/segment/ where Ultralytics may save YOLO weights
        for search_root in [Path("runs/segment"), Path("runs")]:
            yolo_dirs = sorted(search_root.glob(f"**/{model_name}*/weights/best.pt"))
            if yolo_dirs:
                print(f"[checkpoint] Found YOLO checkpoint: {yolo_dirs[-1]}")
                return str(yolo_dirs[-1])
        raise FileNotFoundError(
            f"Checkpoint dir not found: {default_dir}\n"
            "Run training first: python scripts/train.py --config ..."
        )

    # 4. Bare filename (no directory) → look in default checkpoint dir
    candidate = default_dir / p.name
    if candidate.is_file():
        return str(candidate)

    # 4b. Also try with .pt extension for YOLO checkpoints
    if not p.suffix:
        for ext in [".pth", ".pt"]:
            candidate = default_dir / (p.name + ext)
            if candidate.is_file():
                return str(candidate)

    # 5. Nothing matched → helpful error
    _checkpoint_not_found(p, default_dir)


def _best_in_dir(directory: Path, label: str, extensions: tuple = (".pth", ".pt")) -> str:
    """Return the best checkpoint file in the directory."""
    all_ckpts = []
    for ext in extensions:
        all_ckpts.extend(directory.glob(f"*{ext}"))
    all_ckpts = sorted(all_ckpts)
    if not all_ckpts:
        raise FileNotFoundError(
            f"No checkpoint files ({', '.join(extensions)}) found in: {directory}"
        )

    # Try to rank by val_miou value embedded in filename (e.g. val_miou02533)
    def _score(f):
        import re
        # Prefer files with "best" in name
        if "best" in f.stem:
            return 999999
        m = re.search(r"val_miou(\d+)", f.name)
        return int(m.group(1)) if m else 0

    best = max(all_ckpts, key=_score)
    print(f"[checkpoint] Auto-selected best checkpoint from {label}:\n  {best}")
    return str(best)


def _checkpoint_not_found(p: Path, default_dir: Path):
    msg = f"\nCheckpoint not found: {p}\n"
    if default_dir.is_dir():
        pths = list(default_dir.glob("*.pth")) + list(default_dir.glob("*.pt"))
        if pths:
            msg += f"\nAvailable checkpoints in {default_dir}:\n"
            for f in sorted(pths):
                msg += f"  {f.name}\n"
            msg += (
                f"\nRe-run with the correct path, e.g.:\n"
                f"  --checkpoint {pths[-1]}\n"
                f"Or use --checkpoint best  to auto-select the best one."
            )
        else:
            msg += f"No .pth files found in {default_dir}. Run training first."
    else:
        msg += f"Checkpoint directory does not exist: {default_dir}\nRun training first."
    raise FileNotFoundError(msg)


def _save_confusion_matrix(cm: np.ndarray, model_name: str, output_dir: Path) -> Path:
    """Save confusion matrix as a PNG heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cls_names = ["Fiber", "Fragment"]
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = cm.astype(float) / np.where(row_sums == 0, 1, row_sums)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(2))
    ax.set_yticks(range(2))
    ax.set_xticklabels(cls_names)
    ax.set_yticklabels(cls_names)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_title(f"Confusion Matrix — {model_name}")

    thresh = 0.5
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i,
                f"{cm[i, j]}\n({cm_norm[i, j]:.1%})",
                ha="center", va="center", fontsize=10,
                color="white" if cm_norm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    out_path = output_dir / f"{model_name}_confusion_matrix.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to: {out_path}")
    return out_path


def _evaluate_yolo(config: dict, ckpt_path: str, split: str, device: str) -> dict:
    """
    Evaluate a YOLO model:
      1) Ultralytics val() for box/seg mAP
      2) Per-image prediction vs COCO GT for pixel-level IoU, Dice, image IoU, F1
    """
    import time
    import cv2
    from ultralytics import YOLO
    from training.metrics import compute_iou, compute_dice

    data_cfg = config.get("data", {})
    data_yaml = data_cfg.get("yolo_yaml", "data_splits/yolo/dataset.yaml")
    imgsz = config.get("training", {}).get("imgsz", 640)
    cls_names = ["Fiber", "Fragment"]

    model = YOLO(ckpt_path)
    print(f"Loaded YOLO checkpoint: {ckpt_path}")

    params = sum(p.numel() for p in model.model.parameters())
    print(f"Model: yolo26 | Params: {params:,}")

    # ── Part 1: Ultralytics val() for mAP metrics ────────────────────────────
    val_results = model.val(data=data_yaml, imgsz=imgsz, device=device, split=split)

    metrics = {
        "mAP50": float(val_results.box.map50) if hasattr(val_results, "box") else 0.0,
        "mAP75": float(val_results.box.map75) if hasattr(val_results, "box") else 0.0,
        "params": params,
    }
    if hasattr(val_results, "seg"):
        metrics["seg_mAP50"] = float(val_results.seg.map50)
        metrics["seg_mAP75"] = float(val_results.seg.map75)
        metrics["seg_mAP50-95"] = float(val_results.seg.map)

    # ── Part 2: Pixel-level IoU / Dice via per-image inference ────────────────
    # Build GT from COCO annotations for the requested split
    images_dir = data_cfg.get("images_dir", "images/")
    annotation_path = data_cfg.get("annotation", "annotation.json")
    splits_file = data_cfg.get("splits_file", "data_splits/splits.json")

    with open(splits_file) as f:
        splits = json.load(f)
    with open(annotation_path) as f:
        coco = json.load(f)

    split_filenames = set(splits.get(split, []))
    fname_to_img = {img["file_name"]: img for img in coco["images"]}
    img_id_to_anns = {}
    for ann in coco["annotations"]:
        img_id_to_anns.setdefault(ann["image_id"], []).append(ann)

    # Per-class IoU / Dice accumulators
    cls_ious = {1: [], 2: []}
    cls_dices = {1: [], 2: []}
    image_ious = []
    inference_times = []

    # Image-level class sets for F1
    all_pred_cls_sets = []
    all_gt_cls_sets = []

    print(f"\nComputing pixel-level metrics on {len(split_filenames)} {split} images...")

    for fname in sorted(split_filenames):
        img_meta = fname_to_img.get(fname)
        if img_meta is None:
            continue

        img_path = str(Path(images_dir) / fname)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        h_orig, w_orig = img_bgr.shape[:2]

        # Run YOLO inference
        t0 = time.perf_counter()
        preds = model.predict(
            img_path, imgsz=imgsz, device=device,
            conf=0.25, iou=0.45, retina_masks=True, verbose=False,
        )
        t1 = time.perf_counter()
        inference_times.append((t1 - t0) * 1000)

        result = preds[0]

        # Build GT masks at original resolution
        anns = img_id_to_anns.get(img_meta["id"], [])
        gt_masks_list = []  # (mask, class_id) pairs
        for ann in anns:
            mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
            for seg in ann.get("segmentation", []):
                pts = np.array(seg, dtype=np.float32).reshape(-1, 2)
                cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
            gt_masks_list.append((mask, ann["category_id"]))

        gt_cls = set(cid for _, cid in gt_masks_list)
        all_gt_cls_sets.append(gt_cls)

        # Extract predicted masks + labels
        pred_masks_list = []  # (mask, class_id) pairs
        if result.masks is not None and len(result.masks) > 0:
            pred_mask_data = result.masks.data.cpu().numpy()  # (N, mask_h, mask_w)
            pred_classes = result.boxes.cls.cpu().numpy().astype(int)  # 0-indexed

            for k in range(len(pred_mask_data)):
                pm = pred_mask_data[k]
                # Resize to original image size if needed
                if pm.shape[0] != h_orig or pm.shape[1] != w_orig:
                    pm = cv2.resize(pm, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
                pm = (pm > 0.5).astype(np.uint8)
                pred_masks_list.append((pm, pred_classes[k] + 1))  # convert to 1-indexed

        pred_cls = set(cid for _, cid in pred_masks_list)
        all_pred_cls_sets.append(pred_cls)

        # Per-class IoU / Dice: for each GT instance, find best matching prediction
        for gm, g_cls in gt_masks_list:
            best_iou = 0.0
            best_dice = 0.0
            for pm, p_cls in pred_masks_list:
                if p_cls == g_cls:
                    iou_val = compute_iou(pm, gm)
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_dice = compute_dice(pm, gm)
            cls_ious[g_cls].append(best_iou)
            cls_dices[g_cls].append(best_dice)

        # Image-level IoU (class-agnostic union of all masks)
        if gt_masks_list and pred_masks_list:
            gt_union = np.zeros((h_orig, w_orig), dtype=np.uint8)
            for gm, _ in gt_masks_list:
                gt_union = np.maximum(gt_union, gm)
            pred_union = np.zeros((h_orig, w_orig), dtype=np.uint8)
            for pm, _ in pred_masks_list:
                pred_union = np.maximum(pred_union, pm)
            image_ious.append(compute_iou(pred_union, gt_union))

    # Aggregate pixel-level metrics
    iou_per_class = {}
    dice_per_class = {}
    for cls_id in [1, 2]:
        iou_per_class[cls_id] = float(np.mean(cls_ious[cls_id])) if cls_ious[cls_id] else 0.0
        dice_per_class[cls_id] = float(np.mean(cls_dices[cls_id])) if cls_dices[cls_id] else 0.0

    metrics["iou_per_class"] = iou_per_class
    metrics["mIoU"] = float(np.mean(list(iou_per_class.values())))
    metrics["image_iou"] = float(np.mean(image_ious)) if image_ious else 0.0
    metrics["dice_per_class"] = dice_per_class
    metrics["mDice"] = float(np.mean(list(dice_per_class.values())))
    metrics["inference_time_ms"] = float(np.mean(inference_times)) if inference_times else 0.0

    # Image-level F1 / Precision / Recall
    f1_scores, prec_scores, rec_scores = [], [], []
    for c in [1, 2]:
        tp = sum(c in ps and c in gs for ps, gs in zip(all_pred_cls_sets, all_gt_cls_sets))
        fp = sum(c in ps and c not in gs for ps, gs in zip(all_pred_cls_sets, all_gt_cls_sets))
        fn = sum(c not in ps and c in gs for ps, gs in zip(all_pred_cls_sets, all_gt_cls_sets))
        prec = tp / (tp + fp + 1e-6)
        rec = tp / (tp + fn + 1e-6)
        f1 = 2 * prec * rec / (prec + rec + 1e-6)
        name = cls_names[c - 1]
        metrics[f"Precision_{name}"] = float(prec)
        metrics[f"Recall_{name}"] = float(rec)
        metrics[f"F1_{name}"] = float(f1)
        prec_scores.append(prec)
        rec_scores.append(rec)
        f1_scores.append(f1)
    metrics["Precision_macro"] = float(np.mean(prec_scores))
    metrics["Recall_macro"] = float(np.mean(rec_scores))
    metrics["F1_macro"] = float(np.mean(f1_scores))

    # Image-level confusion matrix
    conf_matrix = np.zeros((2, 2), dtype=int)
    for gt_set, pred_set in zip(all_gt_cls_sets, all_pred_cls_sets):
        if not gt_set or not pred_set:
            continue
        gt_c = min(gt_set) - 1
        pred_c = min(pred_set) - 1
        if 0 <= gt_c < 2 and 0 <= pred_c < 2:
            conf_matrix[gt_c][pred_c] += 1
    metrics["confusion_matrix"] = conf_matrix.tolist()

    return metrics


def _print_and_save_results(metrics: dict, model_name: str, split: str, output_arg: str):
    """Print YOLO evaluation results (same format as standard models) and save JSON."""
    cls_names = ["Fiber", "Fragment"]

    iou_pc = metrics.get("iou_per_class", {})
    dice_pc = metrics.get("dice_per_class", {})
    def _iou(c): return float(iou_pc.get(c, iou_pc.get(str(c), 0.0)))
    def _dice(c): return float(dice_pc.get(c, dice_pc.get(str(c), 0.0)))

    print(f"\n{'=' * 60}")
    print(f"Results for {model_name} on {split} split:")
    print(f"{'=' * 60}")
    print(f"  mIoU:            {metrics.get('mIoU', 0.0):.4f}")
    print(f"  IoU Fiber:       {_iou(1):.4f}")
    print(f"  IoU Fragment:    {_iou(2):.4f}")
    print(f"  Image IoU:       {metrics.get('image_iou', 0.0):.4f}")
    print()
    print(f"  mDice:           {metrics.get('mDice', 0.0):.4f}")
    print(f"  Dice Fiber:      {_dice(1):.4f}")
    print(f"  Dice Fragment:   {_dice(2):.4f}")
    print()
    print(f"  mAP50:           {metrics.get('mAP50', 0.0):.4f}")
    print(f"  mAP75:           {metrics.get('mAP75', 0.0):.4f}")
    if "seg_mAP50" in metrics:
        print(f"  Seg mAP50:       {metrics['seg_mAP50']:.4f}")
        print(f"  Seg mAP75:       {metrics['seg_mAP75']:.4f}")
        print(f"  Seg mAP50-95:    {metrics['seg_mAP50-95']:.4f}")
    print()

    print(f"  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*44}")
    for cls in cls_names:
        p = metrics.get(f"Precision_{cls}", 0.0)
        r = metrics.get(f"Recall_{cls}", 0.0)
        f = metrics.get(f"F1_{cls}", 0.0)
        print(f"  {cls:<12} {p:>10.4f} {r:>10.4f} {f:>10.4f}")
    print(f"  {'-'*44}")
    print(f"  {'Macro':<12} {metrics.get('Precision_macro', 0.0):>10.4f}"
          f" {metrics.get('Recall_macro', 0.0):>10.4f}"
          f" {metrics.get('F1_macro', 0.0):>10.4f}")
    print()
    print(f"  Params:          {metrics['params']:,}")
    print(f"  Inference (ms):  {metrics['inference_time_ms']:.1f}")
    print(f"{'=' * 60}")

    # Print confusion matrix if available
    cm = np.array(metrics.get("confusion_matrix", [[0]*2]*2))
    if cm.any():
        print(f"\nConfusion Matrix (rows=GT, cols=Predicted, image-level):")
        header = f"  {'':>12}" + "".join(f"{n:>12}" for n in cls_names)
        print(header)
        print(f"  {'-' * (12 + 12*2)}")
        for i, row_name in enumerate(cls_names):
            row_str = "".join(f"{cm[i, j]:>12}" for j in range(2))
            print(f"  {row_name:<12}{row_str}")
        print()

        # Save confusion matrix image
        results_dir = Path(output_arg).parent if output_arg else Path("outputs/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        _save_confusion_matrix(cm, model_name, results_dir)

    # Save results JSON
    results_dir = Path(output_arg).parent if output_arg else Path("outputs/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_arg or str(results_dir / f"{model_name}_results.json")
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2, default=lambda x: float(x) if hasattr(x, '__float__') else str(x))
    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to .pth file, checkpoint directory, or 'best' to auto-select",
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None, help="Save results JSON to this path")
    args = parser.parse_args()

    config = load_config(args.config)
    model_name = config.get("model", {}).get("name", "model")
    device = args.device or config.get("training", {}).get("device", "cuda")

    # Resolve checkpoint path
    ckpt_path = resolve_checkpoint(args.checkpoint, model_name, config)

    # ── YOLO models use Ultralytics evaluation pipeline ──────────────────────
    if model_name == "yolo26":
        metrics = _evaluate_yolo(config, ckpt_path, args.split, device)
        _print_and_save_results(metrics, model_name, args.split, args.output)
        return

    # Load model
    from scripts.train import get_model
    model = get_model(config)
    state = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
        print(f"Loaded checkpoint: {ckpt_path}  (epoch {state.get('epoch', '?')})")
    else:
        model.load_state_dict(state)
        print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Model: {model_name} | Params: {model.count_parameters():,}")

    # Build test dataloader
    data_cfg = config["data"]
    from data.dataloader import build_dataloader
    test_loader = build_dataloader(
        args.split,
        data_cfg["images_dir"],
        data_cfg["annotation"],
        data_cfg["splits_file"],
        batch_size=1,
        num_workers=2,
        image_size=data_cfg.get("image_size", 640),
    )
    print(f"Evaluating on {args.split} set: {len(test_loader)} batches")

    # Dataset statistics
    ds = test_loader.dataset
    class_ids = data_cfg.get("class_ids", [1, 2])
    class_names = data_cfg.get("class_names", ["Fiber", "Fragment"])
    cls_names_map = dict(zip(class_ids, class_names))
    cls_img_count = {cid: 0 for cid in class_ids}
    for img_meta in ds.images:
        anns = ds._ann_by_image.get(img_meta["id"], [])
        cats_in_img = set(a["category_id"] for a in anns)
        for c in cats_in_img:
            if c in cls_img_count:
                cls_img_count[c] += 1
    total_imgs = len(ds)
    print(f"\n{'=' * 60}")
    print(f"Dataset statistics ({args.split} split, {total_imgs} images):")
    print(f"{'=' * 60}")
    print(f"  {'class_name':<14} {'images':>8} {'pct_of_imgs':>12}")
    print(f"  {'-'*12:14} {'-'*8:>8} {'-'*12:>12}")
    for cid in class_ids:
        cnt = cls_img_count[cid]
        pct = cnt / total_imgs * 100 if total_imgs > 0 else 0.0
        print(f"  {cls_names_map[cid]:<14} {cnt:>8} {pct:>11.2f}%")

    # Evaluate
    from evaluation.evaluator import Evaluator
    evaluator = Evaluator(model, device=device)
    metrics = evaluator.evaluate(test_loader)

    # Print results
    print(f"\n{'=' * 60}")
    print(f"Results for {model_name} on {args.split} split:")
    print(f"{'=' * 60}")
    iou_pc = metrics['iou_per_class']
    dice_pc = metrics.get('dice_per_class', {})
    def _iou(c): return float(iou_pc.get(c, iou_pc.get(str(c), 0.0)))
    def _dice(c): return float(dice_pc.get(c, dice_pc.get(str(c), 0.0)))
    print(f"  mIoU:            {metrics['mIoU']:.4f}")
    print(f"  IoU Fiber:       {_iou(1):.4f}")
    print(f"  IoU Fragment:    {_iou(2):.4f}")
    print(f"  Image IoU:       {metrics.get('image_iou', 0.0):.4f}")
    print()
    print(f"  mDice:           {metrics.get('mDice', 0.0):.4f}")
    print(f"  Dice Fiber:      {_dice(1):.4f}")
    print(f"  Dice Fragment:   {_dice(2):.4f}")
    print()
    print(f"  mAP50:           {metrics['mAP50']:.4f}")
    print(f"  mAP75:           {metrics['mAP75']:.4f}")
    print()
    # Per-class F1 / Precision / Recall table
    cls_names = ["Fiber", "Fragment"]
    print(f"  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*44}")
    for cls in cls_names:
        p = metrics.get(f"Precision_{cls}", 0.0)
        r = metrics.get(f"Recall_{cls}", 0.0)
        f = metrics.get(f"F1_{cls}", 0.0)
        print(f"  {cls:<12} {p:>10.4f} {r:>10.4f} {f:>10.4f}")
    print(f"  {'-'*44}")
    print(f"  {'Macro':<12} {metrics.get('Precision_macro', 0.0):>10.4f}"
          f" {metrics.get('Recall_macro', 0.0):>10.4f}"
          f" {metrics.get('F1_macro', 0.0):>10.4f}")
    print()
    print(f"  Params:          {metrics['params']:,}")
    print(f"  Inference (ms):  {metrics['inference_time_ms']:.1f}")
    print(f"{'=' * 60}")

    # Print confusion matrix
    cm = np.array(metrics.get("confusion_matrix", [[0]*2]*2))
    print(f"\nConfusion Matrix (rows=GT, cols=Predicted, image-level):")
    header = f"  {'':>12}" + "".join(f"{n:>12}" for n in cls_names)
    print(header)
    print(f"  {'-' * (12 + 12*2)}")
    for i, row_name in enumerate(cls_names):
        row_str = "".join(f"{cm[i, j]:>12}" for j in range(2))
        print(f"  {row_name:<12}{row_str}")
    print()

    # Save confusion matrix image
    results_dir = Path(args.output).parent if args.output else Path("outputs/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    _save_confusion_matrix(cm, model_name, results_dir)

    # Save results JSON
    output_path = args.output or str(results_dir / f"{model_name}_results.json")
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2, default=lambda x: float(x) if hasattr(x, '__float__') else str(x))
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
