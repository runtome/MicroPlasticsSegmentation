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


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pth")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None, help="Save results JSON to this path")
    args = parser.parse_args()

    config = load_config(args.config)
    model_name = config.get("model", {}).get("name", "model")
    device = args.device or config.get("training", {}).get("device", "cuda")

    # Load model
    from scripts.train import get_model
    model = get_model(config)
    state = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
        print(f"Loaded checkpoint from epoch {state.get('epoch', '?')}")
    else:
        model.load_state_dict(state)
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
