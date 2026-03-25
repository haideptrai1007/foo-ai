# 1‑Class Audio Command Classifier (MobileNetV3‑Small)

Binary classifier: **“is this audio clip my target command?”** (yes/no), trained on mel‑spectrograms using a MobileNetV3‑Small ImageNet backbone.

This repo supports:

- **Stage A (optional)**: pretrain a **speech‑domain backbone** on `torchaudio.datasets.SPEECHCOMMANDS` (multiclass), then save **backbone‑only** weights.
- **Stage B**: few‑shot fine‑tune a **binary** command detector from your own recorded samples, then export ONNX (and INT8 quantized ONNX).

---

## Project layout

```
command_classifier/
  config.py
  preprocessing/
  model/
  training/
  inference/
  export/
  ui/
pretrained.py   # Stage A backbone pretrain (SPEECHCOMMANDS)
train.py        # Stage B training (folder-based positives/negatives)
test.py         # Stage B inference (checkpoint + audio)
export.py       # Export checkpoint -> ONNX -> INT8 + zip bundle
main.py         # Gradio UI entrypoint (minimal)
requirements.txt
```

---

## Install

### Kaggle

In a Kaggle notebook cell:

```bash
pip install -q -r requirements.txt
```

If Kaggle already provides some packages, installing again is still fine.

---

## Data format (Stage B)

Stage B expects folders of audio files:

```
./data/positive/   # 5–10 clips of your command
./data/negative/   # optional (if absent, code generates synthetic negatives)
```

Supported extensions include: `.wav .mp3 .flac .ogg .m4a .aac .wma`.

---

## Stage A (optional): pretrain backbone on SPEECHCOMMANDS

If your command is **not** one of SPEECHCOMMANDS labels (common), you can still use it to pretrain a backbone on *speech* so the model adapts faster in few‑shot.

This script trains **multiclass** on SPEECHCOMMANDS and saves a **backbone‑only** checkpoint:

```bash
python pretrained.py \
  --out ./pretrained_backbone_sc.pt \
  --epochs 5 \
  --gpus 0 \
  --root ./data/speechcommands \
  --max-train-samples 30000 \
  --max-val-samples 5000
```

It logs per epoch:

- train/val loss
- accuracy
- macro F1

### Output checkpoint format

`pretrained.py` saves:

- `features_state_dict`: MobileNetV3‑Small `.features` weights
- `pretrain_meta`: mel/audio params used during pretraining

---

## Stage B: few‑shot training (binary command detector)

### Train (folder-based)

```bash
python train.py \
  --positive-dir ./data/positive \
  --negative-dir ./data/negative \
  --checkpoint-dir ./checkpoints \
  --gpus 0,1
```

Notes:

- If you only want one GPU, use `--gpus 0`.
- The current Stage B CLI trains from folders; it does **not** yet auto-load the Stage A backbone checkpoint.

### Test a clip

```bash
python test.py \
  --checkpoint ./checkpoints/best_epoch_000.pt \
  --audio ./some_clip.wav \
  --threshold 0.52
```

---

## Export (ONNX + INT8 quant + zip bundle)

```bash
python export.py \
  --checkpoint ./checkpoints/best_epoch_000.pt \
  --export-dir ./export \
  --quantize-mode dynamic
```

Outputs:

- `export/model_fp32.onnx`
- `export/model_int8.onnx`
- `export/command_classifier_export.zip` (contains `model.onnx`, `config.json`, `inference.py`, `requirements_inference.txt`)

---

## Gradio UI (minimal)

Launch the UI:

```bash
python main.py
```

---

## Notes / tips

- **Few-shot works best with real speech negatives** (background speech / “wrong command” speech). Synthetic noise/silence helps, but speech negatives reduce false positives.
- Keep preprocessing consistent across stages: waveform → mel → normalize → resize → ImageNet normalize.

