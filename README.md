# Few-Shot Audio Command Classifier (BCResNet)

Binary classifier: **"is this audio clip my target command?"** (yes/no), trained on log-mel spectrograms using a [BCResNet](https://github.com/Qualcomm-AI-research/bcresnet) backbone (Qualcomm, Interspeech 2021).

Record a few samples of your command, train, test in real-time, then export to INT8 ONNX for edge deployment.

---

## Architecture

- **Model:** BCResNet-1 (tau=1, ~9K params) -- designed for keyword spotting on edge devices
- **Input:** 2-second audio @ 16 kHz -> log-mel spectrogram (1, 40, 251)
- **Output:** binary logit (command / not command)
- **Training:** BCEWithLogitsLoss with dynamic pos_weight, CosineAnnealing LR, AMP
- **Augmentation:** audio-domain (noise, pitch, stretch, time-shift) + SpecAugment (freq/time masking)

Scale up by changing `BCRESNET_TAU` in `config.py`:

| tau | base_c | Params  | Use case          |
|-----|--------|---------|--------------------|
| 1   | 8      | ~9K     | Edge / MCU         |
| 2   | 16     | ~30K    | Mobile             |
| 3   | 24     | ~60K    | Balanced           |
| 6   | 48     | ~200K   | Server / Kaggle    |
| 8   | 64     | ~350K   | Maximum accuracy   |

---

## Project layout

```
command_classifier/
  config.py             # All hyperparameters and paths
  preprocessing/
    audio.py            # Audio loading and resampling
    augmentation.py     # Audio + SpecAugment augmentation
    mel.py              # Mel-spectrogram conversion
    dataset.py          # CommandDataset + create_dataloaders()
  model/
    bcresnet.py         # BCResNet architecture
    subspectralnorm.py  # SubSpectralNorm module
    classifier.py       # build_model() + utilities
  training/
    trainer.py          # CNNTrainer (training loop)
  inference/
    torch_infer.py      # PyTorch inference
    onnx_infer.py       # ONNX Runtime inference
  export/
    quantize.py         # ONNX export + INT8 quantization
    package.py          # Zip bundle packaging
  ui/
    app.py              # Gradio UI (Record -> Train -> Test -> Export)
train.py                # CLI training
test.py                 # CLI inference
export.py               # CLI export
main.py                 # UI entrypoint
```

---

## Install

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `torchaudio`, `gradio`, `soundfile`, `numpy`.

---

## Quick start (Gradio UI)

```bash
python main.py
# or with custom port / public share link
python main.py --port 8080 --share
```

### UI workflow (4 tabs)

| Tab | What to do |
|-----|-----------|
| **1 - Record** | Set target sample count, record clips via microphone, save as Positive |
| **2 - Train** | Set epochs, start training -- live log streams, final metrics shown as JSON |
| **3 - Test** | Record a clip, run inference -- see label, probability, and latency (ms) |
| **4 - Export** | Export checkpoint to FP32 + INT8 ONNX and a deployable zip bundle |

---

## CLI training

```bash
python train.py \
  --positive-dir ./data/positive \
  --negative-dir ./data/negative \
  --checkpoint-dir ./checkpoints \
  --gpus 0
```

---

## CLI inference

```bash
python test.py \
  --checkpoint ./checkpoints/best.pt \
  --audio ./clip.wav \
  --threshold 0.5
```

---

## Export to ONNX

```bash
python export.py \
  --checkpoint ./checkpoints/best.pt \
  --export-dir ./export \
  --quantize-mode dynamic
```

Outputs:
- `export/model_fp32.onnx`
- `export/model_int8.onnx`
- `export/command_classifier_export.zip` -- contains `model.onnx`, `config.json`

---

## Kaggle notebook usage

```python
import sys, os
# Upload the project folder or clone from git
sys.path.insert(0, "/kaggle/working/ai")
os.chdir("/kaggle/working/ai")

from command_classifier.preprocessing.dataset import create_dataloaders
from command_classifier.model.classifier import build_model, prepare_model
from command_classifier.training.trainer import CNNTrainer
from pathlib import Path

# 1. Build dataloaders from your audio folders
train_loader, _, class_weights = create_dataloaders(
    positive_dir="/kaggle/input/your-dataset/positive",
    negative_dir="/kaggle/input/your-dataset/negative",  # optional
)

# 2. Build model
model = build_model(num_classes=1)
model = prepare_model(model)

# 3. Train
trainer = CNNTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=None,
    checkpoint_dir=Path("./checkpoints"),
    freeze_epochs=0,
    pos_weight=class_weights["pos_weight"],
)
result = trainer.train(log_fn=print)
print(result)
```

---

## Edge deployment

The default BCResNet-1 (tau=1) has ~9K parameters and runs well on:
- Raspberry Pi / Jetson Nano
- Android / iOS (via ONNX Runtime Mobile)
- Microcontrollers with >256 KB RAM (INT8 quantized)

Export pipeline: PyTorch -> FP32 ONNX -> INT8 ONNX (dynamic quantization).

Preprocessing on-device: 16 kHz audio -> 256-point FFT -> 40-band log-mel spectrogram. No ImageNet normalization or image resizing required.

---

## Config reference

All hyperparameters are in [`command_classifier/config.py`](command_classifier/config.py):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SAMPLE_RATE` | 16000 | Audio sample rate (Hz) |
| `AUDIO_DURATION_S` | 2.0 | Clip length (seconds) |
| `N_FFT` | 256 | FFT window size |
| `HOP_LENGTH` | 128 | FFT hop length |
| `N_MELS` | 40 | Mel frequency bands |
| `BCRESNET_TAU` | 1 | Model scale (1/2/3/6/8) |
| `AUG_FACTOR` | 5 | Virtual dataset multiplier |
| `BATCH_SIZE` | 64 | Training batch size |
| `NUM_EPOCHS` | 40 | Training epochs |
| `LR` | 1e-4 | Learning rate |

---

## Credits

- BCResNet architecture: [Qualcomm AI Research](https://github.com/Qualcomm-AI-research/bcresnet) (Interspeech 2021)
- SubSpectralNorm: Kim et al., "Broadcasting Residual Learning for Efficient Keyword Spotting"
