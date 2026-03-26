from __future__ import annotations

import copy
import math
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
# torch.amp.GradScaler (new API) was added in PyTorch 2.3;
# torch.amp.autocast was added in PyTorch 1.10.
# Use new API when available, fall back to cuda.amp for older versions.
import torch.amp as _torch_amp
_NEW_AMP = hasattr(_torch_amp, "GradScaler")

from command_classifier.config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    GRAD_CLIP_NORM,
    LR,
    NUM_EPOCHS,
    POS_WEIGHT,
    SCHEDULER,
    VAL_SPLIT,
    WEIGHT_DECAY,
    FREEZE_BACKBONE_EPOCHS,
    UNFREEZE_LR_FACTOR,
    CONFIDENCE_THRESHOLD,
)
from command_classifier.model.classifier import get_param_groups, freeze_backbone, unfreeze_backbone, _unwrap


def _try_import_sklearn_metrics():
    try:
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            f1_score,
            precision_recall_fscore_support,
            roc_auc_score,
            roc_curve,
        )

        return {
            "accuracy_score": accuracy_score,
            "confusion_matrix": confusion_matrix,
            "f1_score": f1_score,
            "precision_recall_fscore_support": precision_recall_fscore_support,
            "roc_auc_score": roc_auc_score,
            "roc_curve": roc_curve,
        }
    except Exception:
        return {}


def _try_import_plotting():
    try:
        import matplotlib.pyplot as plt

        return {"plt": plt}
    except Exception:
        return {}


class CNNTrainer:
    """
    CNNTrainer encapsulates training + evaluation + checkpointing.

    Expected model output:
      - raw logits (shape [batch, 1] or [batch])
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: Optional[torch.device] = None,
        checkpoint_dir: Optional[Path] = None,
        initial_epoch: int = 0,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.initial_epoch = initial_epoch

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.checkpoint_dir = checkpoint_dir or Path("./checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # BCEWithLogitsLoss expects pos_weight as a tensor.
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(POS_WEIGHT)], device=self.device))

        # Freeze backbone initially as per plan
        freeze_backbone(self.model)

        # Optimizer param groups for discriminative learning rates
        # head_lr = LR, backbone_lr = head_lr * UNFREEZE_LR_FACTOR
        param_groups = get_param_groups(self.model, head_lr=LR, backbone_lr_factor=UNFREEZE_LR_FACTOR)
        # Even when frozen, AdamW param groups are fine; frozen params will have grad=None.
        self.optimizer = torch.optim.AdamW(param_groups, lr=LR, weight_decay=WEIGHT_DECAY)

        # AMP
        _enabled = self.device.type == "cuda"
        if _NEW_AMP:
            self.scaler = torch.amp.GradScaler("cuda", enabled=_enabled)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=_enabled)

        self.best_val_loss = float("inf")
        self.best_path: Optional[Path] = None
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
            "val_auc": [],
        }

        self.confidence_threshold = float(CONFIDENCE_THRESHOLD)

    def _compute_metrics(self, probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        metrics_pkg = _try_import_sklearn_metrics()
        if not metrics_pkg:
            # Minimal metric fallback
            preds = (probs >= self.confidence_threshold).astype(np.int64)
            accuracy = float((preds == labels).mean())
            return {"accuracy": accuracy, "precision": 0.0, "recall": 0.0, "f1": 0.0, "auc": 0.0}

        accuracy_score = metrics_pkg["accuracy_score"]
        roc_auc_score = metrics_pkg["roc_auc_score"]
        precision_recall_fscore_support = metrics_pkg["precision_recall_fscore_support"]

        preds = (probs >= self.confidence_threshold).astype(np.int64)

        accuracy = float(accuracy_score(labels, preds))
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
        auc = 0.0
        try:
            auc = float(roc_auc_score(labels, probs))
        except Exception:
            auc = 0.0

        return {
            "accuracy": accuracy,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc": auc,
        }

    def _optimal_threshold(self, probs: np.ndarray, labels: np.ndarray) -> float:
        metrics_pkg = _try_import_sklearn_metrics()
        if not metrics_pkg:
            return self.confidence_threshold

        f1_score = metrics_pkg["f1_score"]
        best_t = self.confidence_threshold
        best_f1 = -1.0

        # Search a small grid; keeps it fast for small datasets.
        for t in np.linspace(0.05, 0.95, 19):
            preds = (probs >= t).astype(np.int64)
            score = float(f1_score(labels, preds, zero_division=0))
            if score > best_f1:
                best_f1 = score
                best_t = float(t)

        return best_t

    def _run_one_epoch(
        self, train: bool, scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> Tuple[float, Dict[str, float]]:
        self.model.train(train)

        total_loss = 0.0
        total = 0

        all_probs: List[float] = []
        all_labels: List[int] = []

        for images, labels in self.train_loader if train else self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True).float()

            if train:
                self.optimizer.zero_grad(set_to_none=True)

            _cuda = self.device.type == "cuda"
            _ctx = torch.amp.autocast("cuda", enabled=_cuda) if _NEW_AMP else torch.cuda.amp.autocast(enabled=_cuda)
            with _ctx:
                logits = self.model(images)
                # logits shape: (batch, 1) typically
                logits = logits.view(-1)
                labels_1d = labels.view(-1)
                loss = self.criterion(logits, labels_1d)

            if train:
                self.scaler.scale(loss).backward()
                if GRAD_CLIP_NORM is not None and GRAD_CLIP_NORM > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP_NORM)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if scheduler is not None:
                    # OneCycleLR is designed to step once per batch.
                    scheduler.step()

            total_loss += float(loss.item()) * images.size(0)
            total += images.size(0)

            probs = torch.sigmoid(logits).detach().float().cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels_1d.detach().float().cpu().numpy().tolist())

        avg_loss = total_loss / max(1, total)

        probs_np = np.array(all_probs, dtype=np.float32)
        labels_np = np.array(all_labels, dtype=np.int64)
        metrics = self._compute_metrics(probs_np, labels_np)

        return avg_loss, metrics

    def _save_checkpoint(self, path: Path, epoch: int, metrics: Dict[str, Any]) -> None:
        payload = {
            "model_state": _unwrap(self.model).state_dict(),
            "epoch": epoch,
            "history": self.history,
            "metrics": metrics,
        }
        torch.save(payload, str(path))

    def train(self, log_fn=None) -> Dict[str, Any]:
        """
        Train for NUM_EPOCHS with frozen backbone first, then unfreeze and fine-tune.

        log_fn: optional callable(str) called after every epoch with a status line.
        """

        def _log(msg: str) -> None:
            if log_fn is not None:
                log_fn(msg)

        device = self.device
        epochs = int(NUM_EPOCHS)

        # Freeze phase
        freeze_epochs = min(int(FREEZE_BACKBONE_EPOCHS), epochs)
        remaining_epochs = max(0, epochs - freeze_epochs)

        global_step = 0
        best_metrics: Dict[str, Any] = {}

        # Scheduler setup: keep per-batch OneCycleLR.
        def make_scheduler(num_epochs: int) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
            if num_epochs <= 0:
                return None
            steps_per_epoch = len(self.train_loader)
            if steps_per_epoch == 0:
                return None
            # OneCycleLR in PyTorch needs max_lr list for multiple param groups; pass base max_lr.
            max_lrs = [group["lr"] for group in self.optimizer.param_groups]
            return torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=max_lrs,
                epochs=num_epochs,
                steps_per_epoch=steps_per_epoch,
                div_factor=10.0,
                final_div_factor=10.0,
                pct_start=0.3,
            )

        scheduler = make_scheduler(freeze_epochs)

        patience_counter = 0

        for epoch in range(self.initial_epoch, freeze_epochs):
            start = time.time()
            train_loss, _ = self._run_one_epoch(train=True, scheduler=scheduler)
            val_loss, val_metrics = self._run_one_epoch(train=False, scheduler=None)
            elapsed = time.time() - start

            self.history["train_loss"].append(float(train_loss))
            self.history["val_loss"].append(float(val_loss))
            self.history["val_accuracy"].append(val_metrics["accuracy"])
            self.history["val_precision"].append(val_metrics["precision"])
            self.history["val_recall"].append(val_metrics["recall"])
            self.history["val_f1"].append(val_metrics["f1"])
            self.history["val_auc"].append(val_metrics["auc"])

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = float(val_loss)
                patience_counter = 0
                best_metrics = val_metrics
                ckpt_path = self.checkpoint_dir / f"best_epoch_{epoch:03d}.pt"
                self._save_checkpoint(ckpt_path, epoch=epoch, metrics=val_metrics)
                self.best_path = ckpt_path
            else:
                patience_counter += 1

            _log(
                f"[frozen] epoch {epoch+1}/{epochs} | "
                f"loss {train_loss:.4f} → val {val_loss:.4f} | "
                f"f1 {val_metrics['f1']:.3f} | "
                f"{elapsed:.1f}s"
                + (" ★" if is_best else f"  (patience {patience_counter}/{EARLY_STOPPING_PATIENCE})")
            )

            if patience_counter >= int(EARLY_STOPPING_PATIENCE):
                _log("Early stopping triggered.")
                break

        # Unfreeze phase
        if remaining_epochs > 0:
            unfreeze_backbone(self.model)
            _log("Backbone unfrozen — fine-tuning all layers.")
            scheduler = make_scheduler(remaining_epochs)
            for i in range(remaining_epochs):
                epoch = freeze_epochs + i
                start = time.time()
                train_loss, _ = self._run_one_epoch(train=True, scheduler=scheduler)
                val_loss, val_metrics = self._run_one_epoch(train=False, scheduler=None)
                elapsed = time.time() - start

                self.history["train_loss"].append(float(train_loss))
                self.history["val_loss"].append(float(val_loss))
                self.history["val_accuracy"].append(val_metrics["accuracy"])
                self.history["val_precision"].append(val_metrics["precision"])
                self.history["val_recall"].append(val_metrics["recall"])
                self.history["val_f1"].append(val_metrics["f1"])
                self.history["val_auc"].append(val_metrics["auc"])

                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = float(val_loss)
                    patience_counter = 0
                    best_metrics = val_metrics
                    ckpt_path = self.checkpoint_dir / f"best_epoch_{epoch:03d}.pt"
                    self._save_checkpoint(ckpt_path, epoch=epoch, metrics=val_metrics)
                    self.best_path = ckpt_path
                else:
                    patience_counter += 1

                _log(
                    f"[tuning] epoch {epoch+1}/{epochs} | "
                    f"loss {train_loss:.4f} → val {val_loss:.4f} | "
                    f"f1 {val_metrics['f1']:.3f} | "
                    f"{elapsed:.1f}s"
                    + (" ★" if is_best else f"  (patience {patience_counter}/{EARLY_STOPPING_PATIENCE})")
                )

                if patience_counter >= int(EARLY_STOPPING_PATIENCE):
                    _log("Early stopping triggered.")
                    break

        # Final best threshold (optional)
        if self.best_path is not None:
            # Load best checkpoint weights into model
            payload = torch.load(str(self.best_path), map_location=self.device)
            _unwrap(self.model).load_state_dict(payload["model_state"])

        # Compute optimal threshold on val set.
        # Collect all probs from val set quickly.
        self.model.eval()
        all_probs: List[float] = []
        all_labels: List[int] = []
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True).float()
                logits = self.model(images).view(-1)
                probs = torch.sigmoid(logits).float().cpu().numpy()
                all_probs.extend(probs.tolist())
                all_labels.extend(labels.view(-1).float().cpu().numpy().tolist())

        probs_np = np.array(all_probs, dtype=np.float32)
        labels_np = np.array(all_labels, dtype=np.int64)
        opt_t = self._optimal_threshold(probs_np, labels_np)
        self.confidence_threshold = opt_t

        return {
            "best_val_loss": self.best_val_loss,
            "best_path": str(self.best_path) if self.best_path is not None else None,
            "best_metrics": best_metrics,
            "optimal_threshold": opt_t,
        }

