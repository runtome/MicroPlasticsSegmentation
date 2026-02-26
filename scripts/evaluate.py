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
        raise FileNotFoundError(
            f"Checkpoint dir not found: {default_dir}\n"
            "Run training first: python scripts/train.py --config ..."
        )

    # 4. Bare filename (no directory) → look in default checkpoint dir
    candidate = default_dir / p.name
    if candidate.is_file():
        return str(candidate)

    # 5. Nothing matched → helpful error
    _checkpoint_not_found(p, default_dir)


def _best_in_dir(directory: Path, label: str) -> str:
    """Return the .pth file with the highest val_miou score in the filename."""
    pths = sorted(directory.glob("*.pth"))
    if not pths:
        raise FileNotFoundError(
            f"No .pth checkpoints found in: {directory}"
        )

    # Try to rank by val_miou value embedded in filename (e.g. val_miou02533)
    def _score(f):
        import re
        m = re.search(r"val_miou(\d+)", f.name)
        return int(m.group(1)) if m else 0

    best = max(pths, key=_score)
    print(f"[checkpoint] Auto-selected best checkpoint from {label}:\n  {best}")
    return str(best)


def _checkpoint_not_found(p: Path, default_dir: Path):
    msg = f"\nCheckpoint not found: {p}\n"
    if default_dir.is_dir():
        pths = list(default_dir.glob("*.pth"))
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

    # Evaluate
    from evaluation.evaluator import Evaluator
    evaluator = Evaluator(model, device=device)
    metrics = evaluator.evaluate(test_loader)

    # Print results
    print(f"\n{'=' * 60}")
    print(f"Results for {model_name} on {args.split} split:")
    print(f"{'=' * 60}")
    print(f"  mIoU:            {metrics['mIoU']:.4f}")
    print(f"  IoU Fiber:       {metrics['iou_per_class'].get(1, 0):.4f}")
    print(f"  IoU Fragment:    {metrics['iou_per_class'].get(2, 0):.4f}")
    print(f"  IoU Film:        {metrics['iou_per_class'].get(3, 0):.4f}")
    print(f"  mAP50:           {metrics['mAP50']:.4f}")
    print(f"  mAP75:           {metrics['mAP75']:.4f}")
    print(f"  F1 macro:        {metrics['F1_macro']:.4f}")
    print(f"  Params:          {metrics['params']:,}")
    print(f"  Inference (ms):  {metrics['inference_time_ms']:.1f}")
    print(f"{'=' * 60}")

    # Save results
    output_path = args.output or f"outputs/results/{model_name}_results.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2, default=lambda x: float(x) if hasattr(x, '__float__') else str(x))
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
