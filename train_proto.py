"""
Prototype training CLI.

Expects one sub-directory per command under --commands-dir:

    data/commands/
        lights_on/   <- 3-5 .wav files
        lights_off/
        play_music/

Usage:
    python train_proto.py \\
        --commands-dir ./data/commands \\
        --output       ./prototype.npz \\
        --method       logmel_delta \\
        --threshold    0.75

Methods:
    logmel          Log-mel mean (40-dim).  Fast, simple.
    logmel_delta    Log-mel + Δ + ΔΔ (120-dim).  Better temporal discrimination.
    pretrained      wav2vec2-base embedding (768-dim).  Best noise robustness.
                    Downloads ~360 MB on first run; cached by torchaudio.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

import torch

from command_classifier.preprocessing.audio import load_audio


_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


def _load_command_waveforms(commands_dir: Path) -> Dict[str, List[torch.Tensor]]:
    result: Dict[str, List[torch.Tensor]] = {}
    for sub in sorted(commands_dir.iterdir()):
        if not sub.is_dir():
            continue
        cmd = sub.name.strip()
        wavs: List[torch.Tensor] = []
        for f in sorted(sub.iterdir()):
            if f.is_file() and f.suffix.lower() in _AUDIO_EXTS:
                try:
                    wavs.append(load_audio(str(f)))
                except Exception as e:
                    print(f"  [skip] {f.name}: {e}")
        if wavs:
            result[cmd] = wavs
            print(f"  {cmd}: {len(wavs)} clip(s)")
        else:
            print(f"  [skip] {cmd}: no valid audio files")
    return result


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="Build prototype classifier from per-command audio folders.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--commands-dir", required=True,
        help="Root directory with one sub-folder per command.",
    )
    ap.add_argument(
        "--output", default="./prototype.npz",
        help="Output .npz path (a .json sidecar is also written).",
    )
    ap.add_argument(
        "--method", default="logmel_delta",
        choices=["logmel", "logmel_delta", "pretrained"],
        help="Embedding method (default: logmel_delta).",
    )
    ap.add_argument(
        "--threshold", type=float, default=0.75,
        help="Rejection threshold: similarity below this → 'none' (default: 0.75).",
    )
    args = ap.parse_args(argv)

    commands_dir = Path(args.commands_dir)
    if not commands_dir.is_dir():
        ap.error(f"--commands-dir not found: {commands_dir}")

    output_path = Path(args.output)

    # ── Load waveforms ─────────────────────────────────────────────────
    print(f"\nLoading audio from: {commands_dir}")
    command_waveforms = _load_command_waveforms(commands_dir)

    if not command_waveforms:
        ap.error("No commands with valid audio found.")
    if len(command_waveforms) < 2:
        print("Warning: only 1 command found — consider adding more for useful classification.")

    # ── Build prototype ────────────────────────────────────────────────
    print(f"\nBuilding '{args.method}' prototypes...")
    if args.method == "logmel":
        from command_classifier.prototype.logmel import LogMelPrototype
        proto = LogMelPrototype()
    elif args.method == "logmel_delta":
        from command_classifier.prototype.logmel_delta import LogMelDeltaPrototype
        proto = LogMelDeltaPrototype()
    else:
        from command_classifier.prototype.pretrained import PretrainedEmbeddingPrototype
        print("  (loading wav2vec2 model — may download ~360 MB on first run)")
        proto = PretrainedEmbeddingPrototype()

    stats = proto.fit(command_waveforms)

    # ── Save ────────────────────────────────────────────────────────────
    from command_classifier.config import (
        AUDIO_DURATION_S, F_MAX, F_MIN, HOP_LENGTH, N_FFT, N_MELS, SAMPLE_RATE,
    )
    audio_config = {
        "sample_rate": SAMPLE_RATE,
        "audio_duration_s": AUDIO_DURATION_S,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "n_mels": N_MELS,
        "f_min": F_MIN,
        "f_max": F_MAX,
    }
    proto.save(
        npz_path=output_path,
        method_name=args.method,
        threshold=args.threshold,
        audio_config=audio_config,
    )

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\nPrototype summary:")
    print(f"  Method:        {args.method}")
    print(f"  Commands:      {list(stats.keys())}")
    print(f"  Embedding dim: {proto.embedding_dim}")
    print(f"  Threshold:     {args.threshold}")
    print(f"\nSaved:")
    print(f"  {output_path}")
    print(f"  {output_path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
