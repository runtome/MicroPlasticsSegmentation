"""
Predictor: single-image and batch inference.
"""
import os
from pathlib import Path
from typing import Dict, Any, List, Union, Optional

import cv2
import numpy as np
import torch
from PIL import Image

from data.transforms import get_inference_transforms


CLASS_NAMES = {0: "Background", 1: "Fiber", 2: "Fragment", 3: "Film"}


class Predictor:
    """
    Runs inference on images using a trained model.
    Supports single image (path or tensor) and batch (directory or list).
    """

    def __init__(self, model, image_size: int = 640, device: str = "cuda"):
        self.model = model
        self.image_size = image_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.transform = get_inference_transforms(image_size)

    @classmethod
    def from_checkpoint(cls, model_class, config: dict, checkpoint_path: str, device: str = "cuda"):
        """Load model from checkpoint."""
        model = model_class(config)
        state = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
        return cls(model, config.get("data", {}).get("image_size", 640), device)

    def preprocess(self, image_input) -> tuple:
        """
        Preprocess image from path, np.ndarray, or PIL.Image.
        Returns (tensor, original_image).
        """
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, np.ndarray):
            img = image_input
            if img.shape[2] == 3 and img.dtype == np.uint8:
                pass  # RGB assumed
        elif isinstance(image_input, Image.Image):
            img = np.array(image_input)
        elif isinstance(image_input, torch.Tensor):
            # Already a tensor — skip transforms
            return image_input, None
        else:
            raise ValueError(f"Unsupported image type: {type(image_input)}")

        original = img.copy()
        transformed = self.transform(image=img)
        tensor = transformed["image"]  # (C, H, W), already normalized by ToTensorV2
        return tensor, original

    @torch.no_grad()
    def predict_single(
        self,
        image_input,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Run inference on a single image.
        Returns prediction dict from model.predict().
        """
        tensor, original = self.preprocess(image_input)
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.to(self.device)

        prediction = self.model.predict(tensor.squeeze(0), threshold=threshold)
        prediction["original_image"] = original
        return prediction

    @torch.no_grad()
    def predict_batch(
        self,
        image_inputs: List,
        threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Run inference on a list of images."""
        results = []
        for img in image_inputs:
            pred = self.predict_single(img, threshold=threshold)
            results.append(pred)
        return results

    @torch.no_grad()
    def predict_directory(
        self,
        images_dir: str,
        output_dir: Optional[str] = None,
        threshold: float = 0.5,
        extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp"),
    ) -> List[Dict[str, Any]]:
        """
        Run inference on all images in a directory.
        Optionally save visualizations to output_dir.
        """
        from inference.visualize import visualize_predictions, save_visualization

        images_dir = Path(images_dir)
        image_paths = [
            p for p in sorted(images_dir.iterdir())
            if p.suffix.lower() in extensions
        ]

        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        results = []
        for img_path in image_paths:
            pred = self.predict_single(img_path, threshold=threshold)
            pred["file_name"] = img_path.name
            results.append(pred)

            if output_dir:
                vis = visualize_predictions(pred["original_image"], pred)
                out_path = Path(output_dir) / img_path.name
                cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        print(f"Processed {len(results)} images from {images_dir}")
        return results
