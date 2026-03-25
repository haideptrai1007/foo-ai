from __future__ import annotations

import argparse
from typing import Optional

from command_classifier.inference.torch_infer import TorchInference


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Run inference with a trained checkpoint (Kaggle CLI).")
    ap.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint (e.g. ./checkpoints/best_epoch_000.pt).")
    ap.add_argument("--audio", required=True, help="Path to an audio file (wav/mp3/etc).")
    ap.add_argument("--threshold", type=float, default=None, help="Optional override for decision threshold.")
    args = ap.parse_args(argv)

    infer = TorchInference(checkpoint_path=args.checkpoint, threshold=args.threshold)
    is_cmd, prob = infer.predict(args.audio)
    print({"is_command": is_cmd, "probability": prob})


if __name__ == "__main__":
    main()

