# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Launch Gradio UI
python main.py
python main.py --share   # public link (Colab/Kaggle)

# CLI: build prototypes from a folder of audio files
python train_proto.py \
  --commands-dir ./data/commands \
  --output       ./prototype.npz \
  --method       logmel_delta \
  --threshold    0.75

# CLI: train BCResNet CNN (optional path)
python train.py --positive-dir ./data/positive --negative-dir ./data/negative --checkpoint-dir ./checkpoints

# CLI: CNN inference / ONNX export
python test.py
python export.py
```

There is no test suite — the project is exercised interactively via `main.py` or the CLI scripts.

## Architecture

**What it does**: Few-shot audio command classifier. Record 3–5 clips per command → compute L2-normalised mean embeddings (prototypes) → classify new audio via cosine similarity with multi-layer rejection.

**Two independent paths**:
1. **Prototype path** (primary): `prototype/` + `train_proto.py` + `main.py`. No training loop, CPU-only, exports to plain numpy arrays.
2. **CNN path** (optional): `model/` + `training/` + `inference/` + `export/`. BCResNet trained on positive/negative clips, exported to ONNX.

### Prototype path data flow

```
load_audio() [preprocessing/audio.py]
  → waveform (1, 32000) @ 16 kHz
  → waveform_to_mel() [preprocessing/mel.py]
  → mel spectrogram (1, 40, T)
  → extract_embedding() [prototype/*.py]
  → n-dim vector
  → BasePrototype.fit() [prototype/base.py]   ← speaker centering + L2-norm
  → cosine similarity + rejection in predict()
  → proto.save() → prototype.npz + prototype.json
```

### Prototype classes (command_classifier/prototype/)

| Class | Dim | Notes |
|---|---|---|
| `LogMelPrototype` | 40 | time-averaged log-mel |
| `LogMelMeanStdPrototype` | 80 | mean + std over time axis (captures spectral variance) |
| `LogMelDeltaPrototype` | 120 | + temporal derivatives Δ, ΔΔ |
| `NearestNeighborPrototype` | 80 | keeps individual clips; per-command score = max sim across clips |
| `PretrainedEmbeddingPrototype` | 768 | wav2vec2-base frozen, layer 6; downloads ~360 MB once |

All share `BasePrototype` (`prototype/base.py`) which handles:
- **Speaker centering**: subtracts the mean of all prototypes before L2-norm (removes voice-signature bias); applied identically at fit and inference time
- **Three-layer rejection**: explicit "other" class wins → `"none"`, then threshold gate, then margin gate (rejects if best − 2nd_best < margin)

### Export format

`proto.save()` writes:
- `prototype.npz` — one numpy array per command (`proto_0000`, …) plus `_speaker_mean`
- `prototype.json` — `method`, `threshold`, `embedding_dim`, `commands` map, `audio` config

On-device inference needs only `numpy`; no PyTorch required.

### Gradio UI (`ui/app.py`, entry point `main.py`)

Four-tab workflow: **Record → Build → Test → Export**. Build tab runs in a background thread and streams log output; Test tab applies an energy gate (silence rejection) before prototype lookup.

### Configuration

All audio parameters live in `command_classifier/config.py` (sample rate 16 kHz, 2 s clips, N_MELS=40, etc.). Change them there — every module imports from config rather than hardcoding values.
