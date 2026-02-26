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


def main():
    parser = argparse.ArgumentParser(description="Run inference on images")
    parser.add_argument("--model", required=True,
                        help="Model name (unet, attention_unet, mask_rcnn, etc.)")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pth")
    parser.add_argument("--input", required=True, help="Input image path or directory")
    parser.add_argument("--output", default="outputs/predictions/",
                        help="Output directory for visualizations")
    parser.add_argument("--config", default=None,
                        help="Config YAML (default: configs/{model}.yaml)")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    # Load config
    config_path = args.config or f"configs/{args.model}.yaml"
    from scripts.train import load_config, get_model
    config = load_config(config_path)
    device = args.device or config.get("training", {}).get("device", "cuda")

    # Build model and load checkpoint
    model = get_model(config)
    import torch
    state = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)

    image_size = config.get("data", {}).get("image_size", 640)

    from inference.predictor import Predictor
    predictor = Predictor(model, image_size=image_size, device=device)

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
