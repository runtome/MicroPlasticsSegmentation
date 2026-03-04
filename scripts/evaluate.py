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

    cls_names = ["Fiber", "Fragment", "Film"]
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = cm.astype(float) / np.where(row_sums == 0, 1, row_sums)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(cls_names)
    ax.set_yticklabels(cls_names)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_title(f"Confusion Matrix — {model_name}")

    thresh = 0.5
    for i in range(3):
        for j in range(3):
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
    """Evaluate a YOLO model using Ultralytics validation pipeline."""
    from models.yolo26.yolo26_wrapper import YOLO26Wrapper

    data_yaml = config.get("data", {}).get("yolo_yaml", "data_splits/yolo/dataset.yaml")
    imgsz = config.get("training", {}).get("imgsz", 640)

    wrapper = YOLO26Wrapper.__new__(YOLO26Wrapper)
    wrapper.config = config
    from ultralytics import YOLO
    wrapper.model = YOLO(ckpt_path)
    print(f"Loaded YOLO checkpoint: {ckpt_path}")

    params = sum(p.numel() for p in wrapper.model.model.parameters())
    print(f"Model: yolo26 | Params: {params:,}")

    results = wrapper.model.val(
        data=data_yaml, imgsz=imgsz, device=device, split=split,
    )

    # Extract per-class metrics from Ultralytics results
    cls_names = ["Fiber", "Fragment", "Film"]
    metrics = {
        "mIoU": 0.0,  # YOLO doesn't directly report IoU; use seg mAP as proxy
        "iou_per_class": {},
        "mAP50": float(results.box.map50) if hasattr(results, "box") else 0.0,
        "mAP75": float(results.box.map75) if hasattr(results, "box") else 0.0,
        "mAP50-95_box": float(results.box.map) if hasattr(results, "box") else 0.0,
        "params": params,
        "inference_time_ms": results.speed.get("inference", 0.0) if hasattr(results, "speed") else 0.0,
    }

    if hasattr(results, "seg"):
        metrics["seg_mAP50"] = float(results.seg.map50)
        metrics["seg_mAP75"] = float(results.seg.map75)
        metrics["seg_mAP50-95"] = float(results.seg.map)

        # Per-class seg AP50
        if hasattr(results.seg, "maps") and results.seg.maps is not None:
            maps = results.seg.maps
            for i, name in enumerate(cls_names):
                if i < len(maps):
                    metrics[f"seg_AP50_{name}"] = float(maps[i])

    if hasattr(results, "box"):
        # Per-class box metrics
        box_p = results.box.p if hasattr(results.box, "p") else []
        box_r = results.box.r if hasattr(results.box, "r") else []
        for i, name in enumerate(cls_names):
            if i < len(box_p):
                metrics[f"Precision_{name}"] = float(box_p[i])
            if i < len(box_r):
                metrics[f"Recall_{name}"] = float(box_r[i])
            p = metrics.get(f"Precision_{name}", 0.0)
            r = metrics.get(f"Recall_{name}", 0.0)
            metrics[f"F1_{name}"] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        # Macro averages
        precs = [metrics.get(f"Precision_{n}", 0.0) for n in cls_names]
        recs = [metrics.get(f"Recall_{n}", 0.0) for n in cls_names]
        f1s = [metrics.get(f"F1_{n}", 0.0) for n in cls_names]
        metrics["Precision_macro"] = sum(precs) / len(precs) if precs else 0.0
        metrics["Recall_macro"] = sum(recs) / len(recs) if recs else 0.0
        metrics["F1_macro"] = sum(f1s) / len(f1s) if f1s else 0.0

    return metrics


def _print_and_save_results(metrics: dict, model_name: str, split: str, output_arg: str):
    """Print YOLO evaluation results and save JSON."""
    cls_names = ["Fiber", "Fragment", "Film"]

    print(f"\n{'=' * 60}")
    print(f"Results for {model_name} on {split} split:")
    print(f"{'=' * 60}")

    print(f"  Box mAP50:       {metrics.get('mAP50', 0.0):.4f}")
    print(f"  Box mAP75:       {metrics.get('mAP75', 0.0):.4f}")
    print(f"  Box mAP50-95:    {metrics.get('mAP50-95_box', 0.0):.4f}")
    print()
    print(f"  Seg mAP50:       {metrics.get('seg_mAP50', 0.0):.4f}")
    print(f"  Seg mAP75:       {metrics.get('seg_mAP75', 0.0):.4f}")
    print(f"  Seg mAP50-95:    {metrics.get('seg_mAP50-95', 0.0):.4f}")
    print()
    for name in cls_names:
        ap = metrics.get(f"seg_AP50_{name}", 0.0)
        print(f"  Seg AP50 {name:<10}: {ap:.4f}")
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
    class_ids = data_cfg.get("class_ids", [1, 2, 3])
    class_names = data_cfg.get("class_names", ["Fiber", "Fragment", "Film"])
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
    print(f"  IoU Film:        {_iou(3):.4f}")
    print(f"  Image IoU:       {metrics.get('image_iou', 0.0):.4f}")
    print()
    print(f"  mDice:           {metrics.get('mDice', 0.0):.4f}")
    print(f"  Dice Fiber:      {_dice(1):.4f}")
    print(f"  Dice Fragment:   {_dice(2):.4f}")
    print(f"  Dice Film:       {_dice(3):.4f}")
    print()
    print(f"  mAP50:           {metrics['mAP50']:.4f}")
    print(f"  mAP75:           {metrics['mAP75']:.4f}")
    print()
    # Per-class F1 / Precision / Recall table
    cls_names = ["Fiber", "Fragment", "Film"]
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
    cm = np.array(metrics.get("confusion_matrix", [[0]*3]*3))
    print(f"\nConfusion Matrix (rows=GT, cols=Predicted, image-level):")
    header = f"  {'':>12}" + "".join(f"{n:>12}" for n in cls_names)
    print(header)
    print(f"  {'-' * (12 + 12*3)}")
    for i, row_name in enumerate(cls_names):
        row_str = "".join(f"{cm[i, j]:>12}" for j in range(3))
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
