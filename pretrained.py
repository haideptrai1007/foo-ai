from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
try:
    from torch.amp import GradScaler, autocast  # PyTorch >= 2.0
except ImportError:  # pragma: no cover
    from torch.cuda.amp import GradScaler, autocast  # type: ignore[no-redef]
from torch.utils.data import DataLoader, Dataset

from command_classifier.config import (
    AUDIO_DURATION_S,
    AUDIO_SAMPLES,
    BATCH_SIZE,
    F_MAX,
    F_MIN,
    HOP_LENGTH,
    LR,
    N_FFT,
    N_MELS,
    NUM_WORKERS,
    SAMPLE_RATE,
    WEIGHT_DECAY,
)
from command_classifier.model.classifier import build_model
from command_classifier.preprocessing.audio import normalize_waveform, pad_or_truncate
from command_classifier.preprocessing.mel import create_mel_transform, mel_to_image, waveform_to_mel


_W = 64  # box width


def _print_header(epochs: int, batch_size: int, lr: float, device: str) -> None:
    print("╔" + "═" * _W + "╗")
    title = "Stage A — SPEECHCOMMANDS Backbone Pretraining"
    print(f"║  {title:<{_W - 2}}║")
    info = f"Epochs: {epochs}   Batch: {batch_size}   LR: {lr:.0e}   Device: {device}"
    print(f"║  {info:<{_W - 2}}║")
    print("╚" + "═" * _W + "╝")
    print()


def _print_epoch(
    epoch: int,
    epochs: int,
    elapsed: float,
    train_loss: float,
    train_metrics: dict,
    val_loss: float,
    val_metrics: dict,
    is_best: bool,
) -> None:
    best_tag = "  ★ new best" if is_best else ""
    print(f"Epoch {epoch:>3} / {epochs}  [{elapsed:.1f}s]")
    print(
        f"  Train │ loss: {train_loss:.4f}  acc: {train_metrics['accuracy'] * 100:.2f}%"
        f"  f1: {train_metrics['f1_macro'] * 100:.2f}%"
    )
    print(
        f"  Val   │ loss: {val_loss:.4f}  acc: {val_metrics['accuracy'] * 100:.2f}%"
        f"  f1: {val_metrics['f1_macro'] * 100:.2f}%{best_tag}"
    )
    print()


def _print_summary(best_epoch: int, best_val_f1: float, out_path: Path) -> None:
    print("═" * (_W + 2))
    print(f"  Best epoch : {best_epoch}")
    print(f"  val_f1_macro: {best_val_f1 * 100:.2f}%")
    print(f"  Saved backbone → {out_path}")
    print("═" * (_W + 2))


def _cleanup_speechcommands_partials(root: Path) -> int:
    """
    Remove stale/incomplete SPEECHCOMMANDS download artifacts.

    torchaudio downloads may leave files like:
      speech_commands_v0.02.tar.gz.<hash>.partial
    when a session is interrupted. Those can break subsequent runs.

    Returns:
        Number of files deleted.
    """

    deleted = 0
    if not root.exists():
        return 0
    for p in root.glob("speech_commands_v0.02.tar.gz*.partial"):
        try:
            p.unlink()
            deleted += 1
        except Exception:
            pass
    return deleted


def _try_sklearn():
    try:
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        return {
            "accuracy_score": accuracy_score,
            "f1_score": f1_score,
            "precision_score": precision_score,
            "recall_score": recall_score,
        }
    except Exception:
        return {}


class SpeechCommandsMelDataset(Dataset):
    """
    SPEECHCOMMANDS -> fixed-length waveform -> mel -> 3x224x224 image.

    Uses the same mel pipeline as the few-shot command detector so the backbone
    transfers cleanly.
    """

    def __init__(self, root: str, subset: str, max_samples: Optional[int] = None, seed: int = 1234) -> None:
        super().__init__()

        try:
            import torchaudio
        except Exception as e:  # pragma: no cover
            raise ImportError("torchaudio is required to use SPEECHCOMMANDS.") from e

        self.root = root
        self.subset = subset
        self.max_samples = max_samples
        self.seed = seed

        self.ds = torchaudio.datasets.SPEECHCOMMANDS(root=root, download=True, subset=subset)

        # Build label mapping
        labels = sorted({self.ds[i][2] for i in range(len(self.ds))})
        self.labels = labels
        self.label_to_idx = {lab: i for i, lab in enumerate(labels)}

        self.mel_transform = create_mel_transform()

        # Optional subsampling for speed
        self.indices = list(range(len(self.ds)))
        if max_samples is not None and max_samples < len(self.indices):
            rng = np.random.default_rng(seed)
            self.indices = rng.choice(self.indices, size=max_samples, replace=False).tolist()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        real_idx = self.indices[idx]
        waveform, sr, label, *_ = self.ds[real_idx]

        # Mono, float32
        if waveform.dim() != 2:
            raise ValueError(f"Unexpected waveform shape: {tuple(waveform.shape)}")
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.to(torch.float32)

        # Resample if needed (torchaudio required; handled by load pipeline normally)
        if sr != SAMPLE_RATE:
            import torchaudio

            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(waveform)

        waveform = pad_or_truncate(waveform, AUDIO_SAMPLES, mode="center")
        waveform = normalize_waveform(waveform)

        mel_db = waveform_to_mel(waveform, self.mel_transform)
        img = mel_to_image(mel_db, image_size=224)
        y = self.label_to_idx[label]
        return img, y


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    sk = _try_sklearn()
    if not sk:
        acc = float((y_true == y_pred).mean())
        return {"accuracy": acc, "f1_macro": 0.0, "precision_macro": 0.0, "recall_macro": 0.0}

    return {
        "accuracy": float(sk["accuracy_score"](y_true, y_pred)),
        "f1_macro": float(sk["f1_score"](y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(sk["precision_score"](y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(sk["recall_score"](y_true, y_pred, average="macro", zero_division=0)),
    }


@torch.no_grad()
def _eval_epoch(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, Dict[str, float]]:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total = 0
    y_true: List[int] = []
    y_pred: List[int] = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += float(loss.item()) * x.size(0)
        total += x.size(0)

        preds = logits.argmax(dim=1)
        y_true.extend(y.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())

    avg_loss = total_loss / max(1, total)
    metrics = _compute_metrics(np.array(y_true), np.array(y_pred))
    return avg_loss, metrics


def _train(
    *,
    root: str,
    out_path: Path,
    epochs: int,
    gpus: str,
    batch_size: int,
    lr: float,
    weight_decay: float,
    max_train_samples: Optional[int],
    max_val_samples: Optional[int],
    num_workers: int,
    seed: int,
    no_data_parallel: bool,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _print_header(epochs=epochs, batch_size=batch_size, lr=lr, device=str(device))

    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    deleted = _cleanup_speechcommands_partials(root_path)
    if deleted > 0:
        print(f"Removed {deleted} stale SPEECHCOMMANDS .partial file(s) under: {root_path}")

    train_ds = SpeechCommandsMelDataset(root=str(root_path), subset="training", max_samples=max_train_samples, seed=seed)
    val_ds = SpeechCommandsMelDataset(root=str(root_path), subset="validation", max_samples=max_val_samples, seed=seed + 1)

    num_classes = len(train_ds.labels)

    model = build_model(pretrained=True, num_classes=num_classes)
    model = model.to(device)

    # Build optimizer BEFORE DataParallel wrapping, then wrap model.
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    use_data_parallel = (not no_data_parallel) and device.type == "cuda" and torch.cuda.device_count() > 1
    if use_data_parallel:
        model = nn.DataParallel(model)
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
    else:
        if device.type == "cuda":
            print(f"Using single GPU (cuda:{torch.cuda.current_device()})")
        else:
            print("Using CPU")
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_epoch = 0
    best_state: Optional[dict] = None

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()

        total_loss = 0.0
        total = 0
        y_true: List[int] = []
        y_pred: List[int] = []

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item()) * x.size(0)
            total += x.size(0)

            preds = logits.argmax(dim=1)
            y_true.extend(y.detach().cpu().numpy().tolist())
            y_pred.extend(preds.detach().cpu().numpy().tolist())

        train_loss = total_loss / max(1, total)
        train_metrics = _compute_metrics(np.array(y_true), np.array(y_pred))

        val_loss, val_metrics = _eval_epoch(model, val_loader, device)

        dt = time.time() - t0
        is_best = val_metrics["f1_macro"] > best_val_f1
        _print_epoch(
            epoch=epoch,
            epochs=epochs,
            elapsed=dt,
            train_loss=train_loss,
            train_metrics=train_metrics,
            val_loss=val_loss,
            val_metrics=val_metrics,
            is_best=is_best,
        )

        if is_best:
            best_val_f1 = float(val_metrics["f1_macro"])
            best_epoch = epoch
            model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            best_state = {k: v.cpu() for k, v in model_state.items()}

    if best_state is None:
        # Defensive fallback: if no epoch improved (unlikely), save final model.
        model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        best_state = {k: v.cpu() for k, v in model_state.items()}
        best_epoch = epochs

    # Save backbone only (features.*) from the best validation epoch.
    backbone = build_model(pretrained=False, num_classes=num_classes)
    backbone.load_state_dict(best_state, strict=True)

    payload = {
        "features_state_dict": backbone.features.state_dict(),
        "best_epoch": best_epoch,
        "best_val_f1_macro": float(best_val_f1),
        "pretrain_meta": {
            "dataset": "SPEECHCOMMANDS",
            "labels": train_ds.labels,
            "sample_rate": SAMPLE_RATE,
            "audio_duration_s": AUDIO_DURATION_S,
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "n_mels": N_MELS,
            "f_min": F_MIN,
            "f_max": F_MAX,
            "image_size": 224,
            "imagenet_mean": [0.485, 0.456, 0.406],
            "imagenet_std": [0.229, 0.224, 0.225],
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(out_path))
    _print_summary(best_epoch=best_epoch, best_val_f1=best_val_f1, out_path=out_path)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Stage A: pretrain backbone on SPEECHCOMMANDS (multiclass).")
    ap.add_argument("--out", required=True, help="Output .pt path (backbone-only checkpoint).")
    ap.add_argument("--epochs", type=int, default=5, help="Number of epochs to train.")
    ap.add_argument("--root", default="./data/speechcommands", help="SPEECHCOMMANDS dataset root/cache directory.")
    ap.add_argument("--gpus", default="0", help="CUDA_VISIBLE_DEVICES (e.g. '0' or '0,1').")
    ap.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    ap.add_argument("--lr", type=float, default=float(LR), help="Learning rate.")
    ap.add_argument("--weight-decay", type=float, default=float(WEIGHT_DECAY), help="Weight decay.")
    ap.add_argument("--num-workers", type=int, default=int(NUM_WORKERS), help="DataLoader workers.")
    ap.add_argument("--seed", type=int, default=1234, help="Random seed.")
    ap.add_argument("--max-train-samples", type=int, default=30000, help="Cap training samples for speed.")
    ap.add_argument("--max-val-samples", type=int, default=5000, help="Cap validation samples for speed.")
    ap.add_argument(
        "--no-data-parallel",
        action="store_true",
        help="Disable DataParallel even when multiple GPUs are visible.",
    )
    args = ap.parse_args(argv)

    _train(
        root=args.root,
        out_path=Path(args.out),
        epochs=int(args.epochs),
        gpus=str(args.gpus),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        max_train_samples=int(args.max_train_samples) if args.max_train_samples and args.max_train_samples > 0 else None,
        max_val_samples=int(args.max_val_samples) if args.max_val_samples and args.max_val_samples > 0 else None,
        num_workers=int(args.num_workers),
        seed=int(args.seed),
        no_data_parallel=bool(args.no_data_parallel),
    )


if __name__ == "__main__":
    main()

