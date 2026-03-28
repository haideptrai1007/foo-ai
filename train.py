from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

from command_classifier.model.classifier import build_model, prepare_model
from command_classifier.preprocessing.dataset import create_dataloaders
from command_classifier.training.trainer import CNNTrainer


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Train the 1-class command classifier (Kaggle CLI).")
    ap.add_argument("--positive-dir", required=True, help="Directory with target-command audio clips.")
    ap.add_argument("--negative-dir", default=None, help="Optional directory with negative clips.")
    ap.add_argument("--checkpoint-dir", default="./checkpoints", help="Output directory for checkpoints.")
    ap.add_argument("--seed", type=int, default=1234, help="Random seed.")
    ap.add_argument("--gpus", default="0,1", help="Value for CUDA_VISIBLE_DEVICES (e.g. '0' or '0,1').")
    args = ap.parse_args(argv)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    positive_dir = Path(args.positive_dir)
    negative_dir = Path(args.negative_dir) if args.negative_dir is not None else None
    checkpoint_dir = Path(args.checkpoint_dir)

    train_loader, val_loader, class_weights = create_dataloaders(
        positive_dir=str(positive_dir),
        negative_dir=str(negative_dir) if negative_dir is not None else None,
        seed=args.seed,
    )

    model = build_model(pretrained=True, num_classes=1)
    model = prepare_model(model)  # wraps with DataParallel when multiple GPUs are visible

    trainer = CNNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=None,
        checkpoint_dir=checkpoint_dir,
        freeze_epochs=0,
        pos_weight=class_weights["pos_weight"],
    )
    result = trainer.train()
    print(result)


if __name__ == "__main__":
    main()

