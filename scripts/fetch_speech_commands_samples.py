"""
One-time script: download SpeechCommands v0.02 and save N samples per class
into command_classifier/data/speech_commands_samples/<class>/sample_N.wav

Run from the repo root:
    python scripts/fetch_speech_commands_samples.py

Requires only stdlib + soundfile (pip install soundfile).
No torchaudio needed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import io
import random
import shutil
import tarfile
import tempfile
import urllib.request
from collections import defaultdict
from pathlib import Path

SPEECHCOMMANDS_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
)

# Folders inside the archive that are metadata, not word classes
_SKIP = {"_background_noise_", "__pycache__"}


def fetch(samples_per_class: int, out_dir: Path, seed: int) -> None:
    random.seed(seed)

    print("Downloading SpeechCommands v0.02 (~2.3 GB) — please wait...")

    with tempfile.TemporaryDirectory() as tmp:
        tar_path = Path(tmp) / "speech_commands.tar.gz"

        # Stream download with progress
        def _reporthook(count, block_size, total):
            mb_done = count * block_size / 1024 / 1024
            mb_total = total / 1024 / 1024
            print(f"\r  {mb_done:.0f} / {mb_total:.0f} MB", end="", flush=True)

        urllib.request.urlretrieve(SPEECHCOMMANDS_URL, tar_path, reporthook=_reporthook)
        print()

        print("Extracting...")
        extract_dir = Path(tmp) / "sc"
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(extract_dir)

        # Collect wav paths grouped by class
        by_class: dict[str, list[Path]] = defaultdict(list)
        for wav in extract_dir.rglob("*.wav"):
            class_name = wav.parent.name
            if class_name in _SKIP:
                continue
            by_class[class_name].append(wav)

        print(f"Found {len(by_class)} classes: {sorted(by_class)}")
        out_dir.mkdir(parents=True, exist_ok=True)

        for class_name, paths in sorted(by_class.items()):
            chosen = random.sample(paths, min(samples_per_class, len(paths)))
            class_dir = out_dir / class_name
            class_dir.mkdir(exist_ok=True)
            for n, src in enumerate(chosen):
                shutil.copy(src, class_dir / f"sample_{n}.wav")
            print(f"  {class_name}: {len(chosen)} samples")

    print(f"\nDone. Samples saved to: {out_dir}")
    print("Commit the data/speech_commands_samples/ directory to the repo.")


def main() -> None:
    from command_classifier.config import SPEECH_COMMANDS_SAMPLES_DIR, SPEECH_COMMANDS_SAMPLES_PER_CLASS

    parser = argparse.ArgumentParser(description="Fetch SpeechCommands samples for bundled negatives.")
    parser.add_argument("--samples", type=int, default=SPEECH_COMMANDS_SAMPLES_PER_CLASS)
    parser.add_argument("--out", type=Path, default=SPEECH_COMMANDS_SAMPLES_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.out.exists():
        print(f"Output dir already exists: {args.out}")
        ans = input("Overwrite? [y/N] ").strip().lower()
        if ans != "y":
            print("Aborted.")
            return
        shutil.rmtree(args.out)

    fetch(samples_per_class=args.samples, out_dir=args.out, seed=args.seed)


if __name__ == "__main__":
    main()
