"""
Universal Trainer: handles train/val loops, 5-fold CV, TensorBoard logging.
"""
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
try:
    from torch.utils.tensorboard import SummaryWriter
    _TENSORBOARD_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    _TENSORBOARD_AVAILABLE = False
    SummaryWriter = None
from tqdm import tqdm

from .callbacks import EarlyStopping, ModelCheckpoint
from .metrics import MetricTracker, compute_iou


class Trainer:
    """
    Universal trainer for all model types.

    Models must implement:
        - train_step(batch, device) -> dict with 'loss' and metric keys
        - val_step(batch, device) -> dict with 'loss', 'iou_per_class', etc.
    """

    def __init__(self, model, config: dict, device: str = "cuda"):
        self.model = model
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Build optimizer
        self.optimizer = self._build_optimizer()
        self.scheduler = None  # built after optimizer

        # Callbacks
        train_cfg = config.get("training", {})
        cb_cfg = config.get("callbacks", {})
        output_cfg = config.get("output", {})

        self.num_epochs = train_cfg.get("num_epochs", 50)
        self.model_name = config.get("model", {}).get("name", "model")
        checkpoint_dir = output_cfg.get("checkpoint_dir", f"outputs/checkpoints/{self.model_name}/")
        log_dir = output_cfg.get("log_dir", f"outputs/logs/{self.model_name}/")

        es_cfg = cb_cfg.get("early_stopping", {})
        self.early_stopping = EarlyStopping(
            patience=es_cfg.get("patience", 10),
            mode="min" if "loss" in es_cfg.get("monitor", "val_loss") else "max",
            verbose=True,
        )
        self.es_monitor = es_cfg.get("monitor", "val_loss")

        ckpt_cfg = cb_cfg.get("checkpoint", {})
        self.checkpoint = ModelCheckpoint(
            checkpoint_dir=checkpoint_dir,
            model_name=self.model_name,
            monitor=ckpt_cfg.get("monitor", "val_miou"),
            mode="max" if "iou" in ckpt_cfg.get("monitor", "val_miou") or "map" in ckpt_cfg.get("monitor", "") else "min",
            save_top_k=ckpt_cfg.get("save_top_k", 1),
        )

        # TensorBoard (optional)
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        if _TENSORBOARD_AVAILABLE and SummaryWriter is not None:
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None

        self.global_step = 0
        self.log_interval = config.get("logging", {}).get("log_interval", 10)

    def _build_optimizer(self):
        train_cfg = self.config.get("training", {})
        opt_name = train_cfg.get("optimizer", "adam").lower()
        lr = train_cfg.get("lr", 1e-3)
        wd = train_cfg.get("weight_decay", 1e-4)

        # Separate backbone/head params if model supports it
        params = self.model.parameters()

        if opt_name == "adam":
            return Adam(params, lr=lr, weight_decay=wd)
        elif opt_name == "adamw":
            return AdamW(params, lr=lr, weight_decay=wd)
        elif opt_name == "sgd":
            momentum = train_cfg.get("momentum", 0.9)
            return SGD(params, lr=lr, momentum=momentum, weight_decay=wd)
        else:
            return Adam(params, lr=lr, weight_decay=wd)

    def _build_scheduler(self):
        cb_cfg = self.config.get("callbacks", {})
        rlrop = cb_cfg.get("reduce_lr", {})
        if rlrop:
            return ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=rlrop.get("factor", 0.5),
                patience=rlrop.get("patience", 5),
                min_lr=rlrop.get("min_lr", 1e-6),
            )
        return None

    def train_epoch(self, loader, epoch: int, fold: Optional[int] = None) -> Dict[str, float]:
        self.model.train()
        tracker = MetricTracker()

        pbar = tqdm(loader, desc=f"Epoch {epoch} [train]")
        for i, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            metrics = self.model.train_step(batch, self.device)
            loss = metrics["loss"]
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Detach for logging
            log_metrics = {k: float(v) for k, v in metrics.items()}
            tracker.update(log_metrics, n=1)

            if i % self.log_interval == 0 and self.writer is not None:
                prefix = f"train/fold{fold}" if fold is not None else "train"
                for k, v in log_metrics.items():
                    self.writer.add_scalar(f"{prefix}/{k}", v, self.global_step)

            self.global_step += 1
            pbar.set_postfix(loss=f"{float(loss):.4f}")

        return tracker.compute()

    @torch.no_grad()
    def val_epoch(self, loader, epoch: int, fold: Optional[int] = None) -> Dict[str, float]:
        self.model.eval()
        tracker = MetricTracker()

        for batch in tqdm(loader, desc=f"Epoch {epoch} [val]"):
            metrics = self.model.val_step(batch, self.device)
            log_metrics = {k: float(v) for k, v in metrics.items()}
            tracker.update(log_metrics, n=1)

        results = tracker.compute()
        if self.writer is not None:
            prefix = f"val/fold{fold}" if fold is not None else "val"
            for k, v in results.items():
                self.writer.add_scalar(f"{prefix}/{k}", v, epoch)

        return results

    def fit(
        self,
        train_loader,
        val_loader,
        fold: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run full training loop."""
        self.scheduler = self._build_scheduler()
        best_metrics = {}

        for epoch in range(1, self.num_epochs + 1):
            train_metrics = self.train_epoch(train_loader, epoch, fold)
            val_metrics = self.val_epoch(val_loader, epoch, fold)

            # Log epoch summary
            train_str = " | ".join(f"{k}={v:.4f}" for k, v in train_metrics.items())
            val_str = " | ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
            print(f"Epoch {epoch}/{self.num_epochs}  Train: {train_str}  Val: {val_str}")

            # Scheduler step
            if self.scheduler is not None:
                val_loss = val_metrics.get("loss", val_metrics.get("val_loss", 0.0))
                self.scheduler.step(val_loss)

            # Checkpoint
            all_metrics = {**{f"train_{k}": v for k, v in train_metrics.items()},
                           **{f"val_{k}": v for k, v in val_metrics.items()}}
            saved = self.checkpoint(self.model, epoch, all_metrics, self.optimizer, fold)
            if saved:
                best_metrics = all_metrics.copy()
                best_metrics["checkpoint_path"] = saved

            # Early stopping
            monitor_val = all_metrics.get(
                self.es_monitor,
                val_metrics.get("loss", val_metrics.get("val_loss", 0.0))
            )
            if self.early_stopping(monitor_val):
                print(f"Early stopping at epoch {epoch}")
                break

        if self.writer is not None:
            self.writer.close()
        return best_metrics

    def fit_kfold(
        self,
        images_dir: str,
        annotation_path: str,
        splits_file: str,
        batch_size: int = 8,
        num_workers: int = 4,
        image_size: int = 640,
    ) -> List[Dict[str, Any]]:
        """Run 5-fold CV."""
        import json
        from data.dataloader import build_dataloader

        with open(splits_file) as f:
            splits = json.load(f)

        num_folds = len(splits["folds"])
        fold_results = []

        for fold in range(num_folds):
            print(f"\n{'='*60}")
            print(f"FOLD {fold + 1}/{num_folds}")
            print(f"{'='*60}")

            # Reset model weights for each fold
            self.model.reset_weights()
            self.model.to(self.device)  # re-move after reset (smp mode re-creates submodules on CPU)
            self.optimizer = self._build_optimizer()
            self.scheduler = None
            self.early_stopping.counter = 0
            self.early_stopping.should_stop = False
            self.early_stopping.best_value = float("inf") if self.early_stopping.mode == "min" else float("-inf")
            self.checkpoint.best_value = float("-inf") if self.checkpoint.mode == "max" else float("inf")
            self.checkpoint.best_path = None

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

            result = self.fit(train_loader, val_loader, fold=fold)
            fold_results.append(result)

        # Summarize
        print("\n5-Fold CV Results:")
        for fold, res in enumerate(fold_results):
            miou = res.get("val_miou", res.get("val_val_miou", "N/A"))
            print(f"  Fold {fold+1}: val_miou={miou}")

        return fold_results
