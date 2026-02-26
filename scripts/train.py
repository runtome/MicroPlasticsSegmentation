"""
Training entry point.

Usage:
    python scripts/train.py --config configs/unet.yaml
    python scripts/train.py --config configs/unet.yaml --fold 0
    python scripts/train.py --config configs/mask_rcnn.yaml
    python scripts/train.py --config configs/yolo26.yaml
    python scripts/train.py --config all               # train every model sequentially
    python scripts/train.py --config all --device cuda # override device for all
"""
import argparse
import os
import sys
import random
import numpy as np
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml


def load_config(config_path: str) -> dict:
    """Load YAML config, merging with base.yaml if 'defaults' key present."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Merge with base config if defaults specified
    if "defaults" in config:
        base_path = Path(config_path).parent / "base.yaml"
        if base_path.exists():
            with open(base_path) as f:
                base = yaml.safe_load(f)
            # Deep merge: config overrides base
            merged = _deep_merge(base, config)
            del merged["defaults"]
            return merged
    return config


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_model(config: dict):
    """Instantiate model by name."""
    model_name = config.get("model", {}).get("name", "unet")
    registry = {
        "unet": ("models.unet.unet", "UNet"),
        "attention_unet": ("models.attention_unet.attention_unet", "AttentionUNet"),
        "dynamic_runext": ("models.dynamic_runext.dynamic_runext", "DynamicRUNext"),
        "mask_rcnn": ("models.mask_rcnn.mask_rcnn_wrapper", "MaskRCNNWrapper"),
        "segformer": ("models.segformer.segformer_wrapper", "SegFormerWrapper"),
        "mask2former": ("models.mask2former.mask2former_wrapper", "Mask2FormerWrapper"),
        "sam2": ("models.sam2.sam2_wrapper", "SAM2Wrapper"),
        "efficient_sam": ("models.efficient_sam.efficient_sam_wrapper", "EfficientSAMWrapper"),
        "rtdetr": ("models.rtdetr.rtdetr_wrapper", "RTDETRWrapper"),
    }
    if model_name not in registry:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(registry.keys())}")

    module_path, class_name = registry[model_name]
    import importlib
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)
    return model_class(config)


def _get_all_configs() -> list:
    """Return all model config paths (everything in configs/ except base.yaml)."""
    configs_dir = Path(__file__).parent.parent / "configs"
    return sorted(
        p for p in configs_dir.glob("*.yaml")
        if p.name != "base.yaml"
    )


def train_one(config_path: str, device: str = None, fold: int = None, resume: str = None):
    """Train a single model config. Returns (model_name, best_metrics or error string)."""
    config = load_config(config_path)
    model_name = config.get("model", {}).get("name", Path(config_path).stem)

    if device:
        config.setdefault("training", {})["device"] = device

    effective_device = config.get("training", {}).get("device", "cuda")
    seed = config.get("training", {}).get("seed", 42)
    set_seed(seed)

    print(f"\nTraining: {model_name}")
    print(f"Device: {effective_device}")
    print(f"Config: {config_path}")

    if model_name == "yolo26":
        _train_yolo(config)
        return model_name, {"status": "completed"}

    model = get_model(config)
    print(f"Parameters: {model.count_parameters():,}")

    if resume:
        state = torch.load(resume, map_location=effective_device)
        model.load_state_dict(state["model_state_dict"])
        print(f"Resumed from {resume}")

    data_cfg = config["data"]
    images_dir = data_cfg["images_dir"]
    annotation_path = data_cfg["annotation"]
    splits_file = data_cfg["splits_file"]
    image_size = data_cfg.get("image_size", 640)
    batch_size = config.get("training", {}).get("batch_size", 8)
    num_workers = 4

    from data.dataloader import build_dataloader
    from training.trainer import Trainer

    trainer = Trainer(model, config, device=effective_device)
    use_cv = config.get("training", {}).get("use_5fold_cv", False)

    if use_cv and fold is None:
        results = trainer.fit_kfold(
            images_dir=images_dir,
            annotation_path=annotation_path,
            splits_file=splits_file,
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=image_size,
        )
        return model_name, results
    else:
        train_loader = build_dataloader(
            "train", images_dir, annotation_path, splits_file,
            batch_size=batch_size, num_workers=num_workers,
            image_size=image_size, fold=fold,
        )
        val_loader = build_dataloader(
            "val", images_dir, annotation_path, splits_file,
            batch_size=batch_size, num_workers=num_workers,
            image_size=image_size, fold=fold,
        )
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        best = trainer.fit(train_loader, val_loader, fold=fold)
        print(f"Best checkpoint: {best.get('checkpoint_path', 'N/A')}")
        return model_name, best


def main():
    parser = argparse.ArgumentParser(description="Train a microplastics segmentation model")
    parser.add_argument(
        "--config", required=True,
        help="Path to config YAML, or 'all' to train every model sequentially",
    )
    parser.add_argument("--fold", type=int, default=None, help="Specific fold for k-fold CV (0-indexed)")
    parser.add_argument("--device", default=None, help="Override device (cuda/cpu)")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint path")
    args = parser.parse_args()

    # ── Train ALL models ─────────────────────────────────────────────────────
    if args.config.lower() == "all":
        configs = _get_all_configs()
        if not configs:
            print("No model configs found in configs/ (excluding base.yaml).")
            return

        print(f"Training {len(configs)} models: {[c.stem for c in configs]}")
        summary = {}
        for cfg_path in configs:
            try:
                name, result = train_one(
                    str(cfg_path), device=args.device,
                    fold=args.fold, resume=args.resume,
                )
                summary[name] = {"status": "ok", "result": result}
            except Exception as e:
                print(f"\n[ERROR] {cfg_path.stem} failed: {e}")
                summary[cfg_path.stem] = {"status": "error", "error": str(e)}

        print("\n" + "=" * 60)
        print("ALL-MODELS TRAINING SUMMARY")
        print("=" * 60)
        for name, info in summary.items():
            if info["status"] == "ok":
                res = info["result"]
                if isinstance(res, dict):
                    miou = res.get("val_miou", res.get("val_val_miou", "N/A"))
                    ckpt = res.get("checkpoint_path", "N/A")
                    print(f"  [OK]    {name:<25} val_miou={miou}  ckpt={ckpt}")
                else:
                    print(f"  [OK]    {name}")
            else:
                print(f"  [FAIL]  {name:<25} {info['error']}")
        return

    # ── Train single model ────────────────────────────────────────────────────
    name, best = train_one(args.config, device=args.device, fold=args.fold, resume=args.resume)
    if isinstance(best, dict):
        print(f"\nBest val_miou: {best.get('val_miou', best.get('val_val_miou', 'N/A'))}")


def _train_yolo(config: dict):
    """Train YOLO models via Ultralytics."""
    from models.yolo26.yolo26_wrapper import YOLO26Wrapper
    yolo_cfg = config.get("data", {})
    data_yaml = yolo_cfg.get("yolo_yaml", "data_splits/yolo/dataset.yaml")
    variants = config.get("model", {}).get("variants", ["yolo11s-seg"])

    results = YOLO26Wrapper.train_all_variants(config, data_yaml, variants)
    print("YOLO training results:", results)


if __name__ == "__main__":
    main()
