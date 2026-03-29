# Few-Shot Audio Command Classifier

Multi-command prototype classifier: **"which command does this audio match?"** (or none).
No training loop — record 3–5 clips per command, compute prototypes, classify in real time.

---

## How it works

1. Record 3–5 clips of each command (e.g. "lights on", "play music", "stop")
2. Extract embeddings with one of three methods (see below)
3. Prototype = L2-normalised mean embedding per command
4. Inference = cosine similarity to each prototype → argmax (or "none" if below threshold)

No backprop, no GPU required. Works in seconds.

---

## Embedding methods

| Method | Dim | Speed | Noise robustness | Notes |
|--------|-----|-------|-----------------|-------|
| `logmel` | 40 | fastest | good | time-averaged log-mel |
| `logmel_delta` | 120 | fast | better | + temporal derivatives (Δ, ΔΔ) |
| `pretrained` | 768 | slow (first run only) | best | wav2vec2-base frozen feature extractor; downloads ~360 MB once |

Start with `logmel_delta` (default). Use `pretrained` if commands sound similar or background noise is high.

---

## Project layout

```
command_classifier/
  config.py             # Audio config and paths
  preprocessing/
    audio.py            # Audio loading, resampling, gradio helpers
    mel.py              # Log-mel spectrogram pipeline
  prototype/
    base.py             # BasePrototype: fit / predict / save / load
    logmel.py           # LogMelPrototype          (40-dim)
    logmel_delta.py     # LogMelDeltaPrototype     (120-dim)
    pretrained.py       # PretrainedEmbeddingPrototype (768-dim, wav2vec2)
  model/
    bcresnet.py         # BCResNet architecture (optional CNN path)
    subspectralnorm.py  # SubSpectralNorm module
    classifier.py       # build_model() utilities
  inference/
    torch_infer.py      # PyTorch inference (CNN path)
    onnx_infer.py       # ONNX Runtime inference
  export/
    quantize.py         # ONNX export + INT8 quantization
    package.py          # Zip bundle packaging
  ui/
    app.py              # Gradio UI (Record -> Build -> Test -> Export)
train_proto.py          # CLI: folder-based prototype training
main.py                 # Gradio UI entrypoint
train.py                # CLI: CNN training (BCResNet path)
test.py                 # CLI: CNN inference
export.py               # CLI: CNN ONNX export
```

---

## Install

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `torchaudio`, `gradio`, `soundfile`, `numpy`.

---

## Quick start — Gradio UI

```bash
python main.py
# with public share link (Kaggle / Colab)
python main.py --share
```

### UI workflow (4 tabs)

| Tab | What to do |
|-----|-----------|
| **1 · Record** | Type a command name, record clips with the microphone, click **Add Clip** — repeat per command |
| **2 · Build** | Choose embedding method + threshold, click **Build Prototypes** |
| **3 · Test** | Record a clip, click **Identify Command** — see best match + similarity scores |
| **4 · Export** | Save `.npz` + `.json` bundle for on-device inference |

---

## CLI usage — `train_proto.py`

Organise your audio files into one folder per command:

```
data/commands/
    lights_on/    ← 3–5 .wav files
    lights_off/
    play_music/
    stop/
```

Then build prototypes:

```bash
python train_proto.py \
  --commands-dir ./data/commands \
  --output       ./prototype.npz \
  --method       logmel_delta \
  --threshold    0.75
```

Outputs:
- `prototype.npz` — numpy prototype arrays
- `prototype.json` — method, commands, threshold, audio config

Supported audio formats: `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, `.aac`

---

## Kaggle notebook usage

```python
import sys, os
sys.path.insert(0, "/kaggle/working/ai")
os.chdir("/kaggle/working/ai")

from pathlib import Path
from command_classifier.preprocessing.audio import load_audio
from command_classifier.prototype.logmel_delta import LogMelDeltaPrototype

# 1. Load waveforms per command from your dataset folders
command_waveforms = {}
commands_dir = Path("/kaggle/input/your-dataset/commands")
for sub in sorted(commands_dir.iterdir()):
    if sub.is_dir():
        wavs = [load_audio(str(f)) for f in sorted(sub.glob("*.wav"))]
        if wavs:
            command_waveforms[sub.name] = wavs
            print(f"  {sub.name}: {len(wavs)} clip(s)")

# 2. Build prototype
proto = LogMelDeltaPrototype()
stats = proto.fit(command_waveforms)
print("Commands:", proto.commands)
print("Embedding dim:", proto.embedding_dim)

# 3. Run inference on a new clip
waveform = load_audio("/kaggle/working/test_clip.wav")
best_cmd, best_sim, all_sims = proto.predict(waveform, threshold=0.75)
print(f"Detected: {best_cmd}  (similarity={best_sim:.3f})")
print("All scores:", all_sims)

# 4. Save for deployment
from command_classifier.config import (
    SAMPLE_RATE, AUDIO_DURATION_S, N_FFT, HOP_LENGTH, N_MELS, F_MIN, F_MAX,
)
proto.save(
    npz_path=Path("/kaggle/working/prototype.npz"),
    method_name="logmel_delta",
    threshold=0.75,
    audio_config={
        "sample_rate": SAMPLE_RATE,
        "audio_duration_s": AUDIO_DURATION_S,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "n_mels": N_MELS,
        "f_min": F_MIN,
        "f_max": F_MAX,
    },
)
```

Or use the CLI directly in a Kaggle notebook cell:

```bash
!python train_proto.py \
  --commands-dir /kaggle/input/your-dataset/commands \
  --output       /kaggle/working/prototype.npz \
  --method       logmel_delta \
  --threshold    0.75
```

### Launch the UI from Kaggle

```python
import subprocess
subprocess.Popen(["python", "main.py", "--share"])
# Gradio will print a public URL — open it in your browser
```

---

## On-device inference (no PyTorch)

At inference time only `numpy` + your mel pipeline is needed — the `.npz` file contains plain arrays.

```python
import numpy as np
import json

# Load prototypes
data = np.load("prototype.npz")
meta = json.load(open("prototype.json"))
key_map = meta["commands"]          # {"proto_0000": "lights_on", ...}
threshold = meta["threshold"]       # 0.75

prototypes = {key_map[k]: data[k] for k in data.files}

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def classify(embedding, prototypes, threshold):
    sims = {cmd: cosine_sim(embedding, proto) for cmd, proto in prototypes.items()}
    best = max(sims, key=sims.get)
    return (best, sims[best]) if sims[best] >= threshold else ("none", sims[best])
```

---

## Export format

`proto.save()` writes two files:

| File | Contents |
|------|---------|
| `prototype.npz` | numpy arrays, one per command (`proto_0000`, `proto_0001`, …) |
| `prototype.json` | `method`, `threshold`, `embedding_dim`, `commands` map, `audio` config |

Example `prototype.json`:

```json
{
  "method": "logmel_delta",
  "threshold": 0.75,
  "embedding_dim": 120,
  "commands": {
    "proto_0000": "lights_on",
    "proto_0001": "lights_off",
    "proto_0002": "play_music"
  },
  "audio": {
    "sample_rate": 16000,
    "audio_duration_s": 2.0,
    "n_fft": 256,
    "hop_length": 128,
    "n_mels": 40,
    "f_min": 0,
    "f_max": 8000
  }
}
```

---

## Config reference

All audio parameters are in [`command_classifier/config.py`](command_classifier/config.py):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SAMPLE_RATE` | 16000 | Audio sample rate (Hz) |
| `AUDIO_DURATION_S` | 2.0 | Clip length (seconds) |
| `N_FFT` | 256 | FFT window size |
| `HOP_LENGTH` | 128 | FFT hop |
| `N_MELS` | 40 | Mel frequency bands |
| `F_MIN` | 0 | Min mel frequency |
| `F_MAX` | 8000 | Max mel frequency |

---

## BCResNet (optional CNN path)

If you want to go beyond few-shot prototypes and train a full CNN, the BCResNet backbone is included.

Scale by changing `BCRESNET_TAU` in `config.py`:

| tau | base_c | Params  | Use case          |
|-----|--------|---------|--------------------|
| 1   | 8      | ~9K     | Edge / MCU         |
| 2   | 16     | ~30K    | Mobile             |
| 3   | 24     | ~60K    | Balanced           |
| 6   | 48     | ~200K   | Server / Kaggle    |
| 8   | 64     | ~350K   | Maximum accuracy   |

```bash
python train.py \
  --positive-dir ./data/positive \
  --negative-dir ./data/negative \
  --checkpoint-dir ./checkpoints \
  --gpus 0
```

---

## Credits

- BCResNet architecture: [Qualcomm AI Research](https://github.com/Qualcomm-AI-research/bcresnet) (Interspeech 2021)
- SubSpectralNorm: Kim et al., "Broadcasting Residual Learning for Efficient Keyword Spotting"
