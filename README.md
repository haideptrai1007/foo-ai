# 1‑Class Audio Command Classifier (MobileNetV3‑Small)

Binary classifier: **"is this audio clip my target command?"** (yes/no), trained on mel‑spectrograms using a MobileNetV3‑Small backbone.

Two stages:

- **Stage A (optional)**: pretrain a speech‑domain backbone on `SPEECHCOMMANDS` (multiclass), save backbone‑only weights.
- **Stage B**: few‑shot binary command detector — record samples in the UI, train, test in real-time, then export to ONNX.

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
  ui/app.py       # Gradio few-shot UI
pretrained.py     # Stage A backbone pretrain (SPEECHCOMMANDS)
train.py          # Stage B training (folder-based CLI)
test.py           # Stage B inference (CLI)
export.py         # Export checkpoint -> ONNX -> INT8 + zip bundle
main.py           # UI entrypoint
requirements.txt
```

---

## Install

```bash
pip install -r requirements.txt
```

---

## Stage A (optional): Pretrain backbone on SPEECHCOMMANDS

Pretrains a multiclass backbone on speech audio so it transfers better to your command.

```bash
python pretrained.py \
  --out ./pretrained_backbone.pt \
  --epochs 5 \
  --gpus 0,1 \
  --root ./data/speechcommands \
  --max-train-samples 30000 \
  --max-val-samples 5000
```

**Structured training output:**
```
╔════════════════════════════════════════════════════════════════╗
║  Stage A — SPEECHCOMMANDS Backbone Pretraining                 ║
║  Epochs: 5   Batch: 128   LR: 1e-03   Device: cuda:0          ║
╚════════════════════════════════════════════════════════════════╝

Epoch   1 / 5  [12.3s]
  Train │ loss: 0.4321  acc: 87.65%  f1: 84.32%
  Val   │ loss: 0.3210  acc: 90.12%  f1: 89.01%  ★ new best

...

══════════════════════════════════════════════════════════════════
  Best epoch : 4
  val_f1_macro: 92.34%
  Saved backbone → pretrained_backbone.pt
══════════════════════════════════════════════════════════════════
```

Saves:
- `features_state_dict` — MobileNetV3‑Small `.features` weights
- `pretrain_meta` — mel/audio config used during pretraining

Flags:
- `--no-data-parallel` — disable DataParallel even when multiple GPUs are visible

---

## Stage B: Few-shot training via Gradio UI

### Launch

```bash
# Default (raw ImageNet MobileNetV3 backbone)
python main.py

# With Stage A pretrained backbone
python main.py --ckpt ./pretrained_backbone.pt

# Custom port / public share link
python main.py --ckpt ./pretrained_backbone.pt --port 8080 --share
```

### UI workflow (4 tabs)

| Tab | What to do |
|-----|-----------|
| **1 · Record** | Set samples-per-class target, record via microphone, save as Positive or Negative |
| **2 · Train** | Set epochs, start training — live log streams as it runs, final metrics shown as JSON |
| **3 · Test** | Record a clip, run inference — see label, probability, and inference time (ms) |
| **4 · Export** | Export trained checkpoint to FP32 + INT8 ONNX and a deployable zip bundle |

### Folder-based CLI training (alternative)

```bash
python train.py \
  --positive-dir ./data/positive \
  --negative-dir ./data/negative \
  --checkpoint-dir ./checkpoints \
  --gpus 0,1
```

### CLI inference

```bash
python test.py \
  --checkpoint ./checkpoints/best.pt \
  --audio ./clip.wav \
  --threshold 0.52
```

---

## Export (CLI)

```bash
python export.py \
  --checkpoint ./checkpoints/best.pt \
  --export-dir ./export \
  --quantize-mode dynamic
```

Outputs:
- `export/model_fp32.onnx`
- `export/model_int8.onnx`
- `export/command_classifier_export.zip` — contains `model.onnx`, `config.json`, `inference.py`

---

## Tips

- **Few-shot works best with real speech negatives.** Synthetic noise/silence helps, but background speech reduces false positives significantly.
- Keep preprocessing consistent: waveform → mel → normalize → 224×224 → ImageNet normalize.
- For best results with Stage A, use a GPU; the pretrain step is the slowest part.
