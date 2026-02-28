"""
Run inference on new images.

Usage:
    python scripts/predict.py --model unet --checkpoint outputs/checkpoints/unet/best.pth --input images/
    python scripts/predict.py --model unet --checkpoint best.pth --input path/to/image.jpg --output outputs/predictions/
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml


def _predict_split(predictor, config, args):
    """
    Run inference on a full data split and save side-by-side GT vs prediction images.
    Each saved image: [GT masks + labels | Predicted mask + class]
    """
    import json
    import numpy as np
    import torch
    import cv2
    from data.dataset import MicroPlasticsDataset
    from data.transforms import get_val_transforms
    from inference.visualize import visualize_gt, visualize_predictions, make_side_by_side

    data_cfg = config["data"]
    image_size = data_cfg.get("image_size", 640)

    with open(data_cfg["splits_file"]) as f:
        splits = json.load(f)
    file_names = splits.get(args.split, [])
    if not file_names:
        print(f"No images found for split '{args.split}' in {data_cfg['splits_file']}")
        return

    dataset = MicroPlasticsDataset(
        images_dir=data_cfg["images_dir"],
        annotation_path=data_cfg["annotation"],
        file_names=file_names,
        transforms=get_val_transforms(image_size),
        image_size=image_size,
    )

    output_dir = Path(args.output) / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Running inference on '{args.split}' split: {len(dataset)} images → {output_dir}")

    # ImageNet denorm constants
    _mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    for item in dataset:
        img_tensor  = item["image"].to(predictor.device)   # (C, H, W) normalized
        gt_masks    = item["masks"].numpy()                # (N, H, W)
        gt_labels   = item["labels"].numpy()               # (N,)
        file_name   = item["file_name"]

        # Denormalize tensor → uint8 RGB for visualization
        img_vis = img_tensor.cpu().numpy().transpose(1, 2, 0)
        img_vis = (img_vis * _std + _mean).clip(0.0, 1.0)
        img_vis = (img_vis * 255).astype(np.uint8)

        # Predict
        with torch.no_grad():
            pred = predictor.model.predict(img_tensor, threshold=args.threshold)

        # Build GT and prediction panels, then join side-by-side
        n_inst = len(gt_labels)
        gt_title   = f"GT  ({n_inst} instance{'s' if n_inst != 1 else ''})"
        pred_title = "Pred"
        if "predicted_class" in pred:
            from inference.predictor import CLASS_NAMES
            pred_title = f"Pred  {CLASS_NAMES.get(pred['predicted_class'], pred['predicted_class'])}"

        gt_panel   = visualize_gt(img_vis, gt_masks, gt_labels)
        pred_panel = visualize_predictions(img_vis.copy(), pred)
        canvas     = make_side_by_side(gt_panel, pred_panel, gt_title, pred_title)

        out_path = output_dir / Path(file_name).name
        cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    print(f"Saved {len(dataset)} side-by-side images to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run inference on images")
    parser.add_argument("--model", required=True,
                        help="Model name (unet, attention_unet, mask_rcnn, etc.)")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pth")
    parser.add_argument("--input", default=None, help="Input image path or directory (required when --split is not used)")
    parser.add_argument("--output", default="outputs/predictions/",
                        help="Output directory for visualizations")
    parser.add_argument("--config", default=None,
                        help="Config YAML (default: configs/{model}.yaml)")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--split", default=None, choices=["train", "val", "test"],
        help="Run on a full data split with GT comparison (saves side-by-side images)",
    )
    args = parser.parse_args()

    if not args.split and not args.input:
        parser.error("--input is required when --split is not specified")

    # Load config
    config_path = args.config or f"configs/{args.model}.yaml"
    from scripts.train import load_config, get_model
    config = load_config(config_path)
    device = args.device or config.get("training", {}).get("device", "cuda")

    # Resolve checkpoint path (same logic as evaluate.py)
    import torch
    from scripts.evaluate import resolve_checkpoint
    model_name = config.get("model", {}).get("name", args.model)
    ckpt_path = resolve_checkpoint(args.checkpoint, model_name, config)
    print(f"Loading checkpoint: {ckpt_path}")

    # Build model and load checkpoint
    model = get_model(config)
    state = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)

    image_size = config.get("data", {}).get("image_size", 640)

    from inference.predictor import Predictor
    predictor = Predictor(model, image_size=image_size, device=device)

    # --- Split mode: run on full train/val/test with GT side-by-side ---
    if args.split:
        _predict_split(predictor, config, args)
        return

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir():
        print(f"Running inference on directory: {input_path}")
        results = predictor.predict_directory(
            str(input_path),
            output_dir=str(output_dir),
            threshold=args.threshold,
        )
        print(f"Saved {len(results)} visualizations to {output_dir}")
    elif input_path.is_file():
        print(f"Running inference on: {input_path}")
        result = predictor.predict_single(str(input_path), threshold=args.threshold)

        # Save visualization
        import cv2
        from inference.visualize import visualize_predictions
        vis = visualize_predictions(result.get("original_image"), result)
        out_path = output_dir / input_path.name
        cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"Saved visualization to {out_path}")

        # Print prediction summary
        if "predicted_class" in result:
            from inference.predictor import CLASS_NAMES
            print(f"Predicted class: {CLASS_NAMES.get(result['predicted_class'], result['predicted_class'])}")
        elif "labels" in result:
            print(f"Detected {len(result['labels'])} instances")
    else:
        print(f"Input not found: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
