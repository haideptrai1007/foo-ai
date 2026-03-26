from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from command_classifier.config import CHECKPOINT_DIR, EXPORT_DIR


def _load_gradio():
    try:
        import gradio as gr

        return gr
    except Exception as e:  # pragma: no cover
        raise ImportError("gradio is required for the UI.") from e


def _save_clips_to_dir(clips: List[Tuple[int, np.ndarray]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (sr, arr) in enumerate(clips):
        if arr.ndim > 1:
            arr = arr.mean(axis=-1)
        arr = arr.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 32768.0
        sf.write(str(out_dir / f"clip_{i:04d}.wav"), arr, sr)


def _load_backbone(model, backbone_ckpt: Optional[str]) -> None:
    """Load Stage A backbone weights into model.features if a checkpoint is given."""
    if not backbone_ckpt:
        return
    import torch

    payload = torch.load(backbone_ckpt, map_location="cpu")
    if "features_state_dict" in payload:
        model.features.load_state_dict(payload["features_state_dict"], strict=True)
    elif "model_state" in payload:
        model.load_state_dict(payload["model_state"], strict=False)


def create_app(backbone_ckpt: Optional[str] = None):
    gr = _load_gradio()

    # Shared state: list of (sr, np.ndarray) tuples per class
    init_state: Dict[str, List] = {"positive": [], "negative": []}

    with gr.Blocks(title="Few-Shot Audio Command Classifier") as demo:
        gr.Markdown("# Few-Shot Audio Command Classifier")
        gr.Markdown("Record samples → Train → Test in real-time → Export")

        clips_state = gr.State(value={"positive": [], "negative": []})

        # ── Tab 1: Record ────────────────────────────────────────────────────
        with gr.Tab("1 · Record"):
            gr.Markdown("Record audio clips for each class. Use the microphone widget, then save.")

            with gr.Row():
                n_samples = gr.Number(label="Target samples per class", value=10, precision=0)

            mic_audio = gr.Audio(sources=["microphone"], type="numpy", label="Microphone")

            with gr.Row():
                btn_pos = gr.Button("Save as Positive", variant="primary")
                btn_neg = gr.Button("Save as Negative")
                btn_clear = gr.Button("Clear All", variant="stop")

            counter_display = gr.Textbox(
                label="Recorded so far",
                value="Positive: 0   Negative: 0",
                interactive=False,
            )

            def _counter_text(state):
                return f"Positive: {len(state['positive'])}   Negative: {len(state['negative'])}"

            def save_positive(audio, state):
                if audio is None:
                    return state, _counter_text(state)
                sr, arr = audio
                state["positive"].append((sr, arr.copy()))
                return state, _counter_text(state)

            def save_negative(audio, state):
                if audio is None:
                    return state, _counter_text(state)
                sr, arr = audio
                state["negative"].append((sr, arr.copy()))
                return state, _counter_text(state)

            def clear_all(_state):
                new_state = {"positive": [], "negative": []}
                return new_state, _counter_text(new_state)

            btn_pos.click(fn=save_positive, inputs=[mic_audio, clips_state], outputs=[clips_state, counter_display])
            btn_neg.click(fn=save_negative, inputs=[mic_audio, clips_state], outputs=[clips_state, counter_display])
            btn_clear.click(fn=clear_all, inputs=[clips_state], outputs=[clips_state, counter_display])

        # ── Tab 2: Train ─────────────────────────────────────────────────────
        with gr.Tab("2 · Train"):
            gr.Markdown("Train the few-shot classifier on your recorded samples.")

            with gr.Row():
                train_epochs = gr.Number(label="Epochs", value=20, precision=0)
                ckpt_dir_box = gr.Textbox(label="Checkpoint output dir", value=str(CHECKPOINT_DIR))

            if backbone_ckpt:
                gr.Markdown(f"**Backbone:** `{backbone_ckpt}`")
            else:
                gr.Markdown("**Backbone:** ImageNet MobileNetV3 (no Stage A checkpoint)")

            train_btn = gr.Button("Start Training", variant="primary")
            train_log = gr.Textbox(label="Training log", lines=12, interactive=False)
            train_result = gr.JSON(label="Final metrics")

            def on_train(state, epochs, ckpt_dir):
                import torch
                from command_classifier.config import NUM_EPOCHS
                from command_classifier.model.classifier import build_model, prepare_model
                from command_classifier.preprocessing.dataset import create_dataloaders
                from command_classifier.training.trainer import CNNTrainer

                pos_clips = state.get("positive", [])
                neg_clips = state.get("negative", [])

                if len(pos_clips) == 0:
                    yield "No positive samples recorded.", None
                    return

                logs = []

                def emit(msg):
                    logs.append(msg)
                    return "\n".join(logs)

                yield emit("Saving clips to temp directory..."), None

                with tempfile.TemporaryDirectory() as tmp:
                    pos_dir = Path(tmp) / "positive"
                    neg_dir = Path(tmp) / "negative" if neg_clips else None
                    _save_clips_to_dir(pos_clips, pos_dir)
                    if neg_clips and neg_dir is not None:
                        _save_clips_to_dir(neg_clips, neg_dir)

                    yield emit(f"Saved {len(pos_clips)} positive, {len(neg_clips)} negative clips."), None
                    yield emit("Building dataloaders..."), None

                    train_loader, val_loader, _ = create_dataloaders(
                        positive_dir=str(pos_dir),
                        negative_dir=str(neg_dir) if neg_dir else None,
                        seed=1234,
                    )

                    yield emit(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}"), None
                    yield emit("Building model..."), None

                    model = build_model(pretrained=True, num_classes=1)
                    _load_backbone(model, backbone_ckpt)
                    model = prepare_model(model)

                    yield emit("Starting training..."), None

                    # Temporarily override NUM_EPOCHS via a monkey-patch-safe approach
                    import command_classifier.training.trainer as _trainer_mod
                    _orig = _trainer_mod.NUM_EPOCHS
                    _trainer_mod.NUM_EPOCHS = int(epochs)

                    try:
                        trainer = CNNTrainer(
                            model=model,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            device=None,
                            checkpoint_dir=Path(ckpt_dir),
                        )
                        metrics = trainer.train()
                    finally:
                        _trainer_mod.NUM_EPOCHS = _orig

                yield emit("Training complete."), metrics

            train_btn.click(
                fn=on_train,
                inputs=[clips_state, train_epochs, ckpt_dir_box],
                outputs=[train_log, train_result],
            )

        # ── Tab 3: Test (real-time) ──────────────────────────────────────────
        with gr.Tab("3 · Test"):
            gr.Markdown("Record a clip and run inference. Inference time and probability are shown.")

            test_ckpt = gr.Textbox(label="Checkpoint (.pt) path", value=str(CHECKPOINT_DIR / "best.pt"))
            test_threshold = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="Threshold")
            test_audio = gr.Audio(sources=["microphone"], type="numpy", label="Microphone")
            test_btn = gr.Button("Run Inference", variant="primary")

            with gr.Row():
                test_result = gr.Textbox(label="Result", interactive=False)
                test_prob = gr.Number(label="Probability", interactive=False)
                test_latency = gr.Number(label="Inference time (ms)", interactive=False)

            def on_test(ckpt, threshold, audio):
                if audio is None:
                    return "No audio recorded.", 0.0, 0.0
                from command_classifier.inference.torch_infer import TorchInference

                infer = TorchInference(checkpoint_path=ckpt, threshold=threshold)
                t0 = time.perf_counter()
                is_cmd, prob = infer.predict(audio)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                label = "COMMAND DETECTED" if is_cmd else "not detected"
                return label, round(prob, 4), round(latency_ms, 1)

            test_btn.click(
                fn=on_test,
                inputs=[test_ckpt, test_threshold, test_audio],
                outputs=[test_result, test_prob, test_latency],
            )

        # ── Tab 4: Export ────────────────────────────────────────────────────
        with gr.Tab("4 · Export"):
            gr.Markdown("Export trained model to INT8 ONNX and package as a deployable zip.")

            export_ckpt = gr.Textbox(label="Checkpoint (.pt) path", value=str(CHECKPOINT_DIR / "best.pt"))
            export_zip = gr.Textbox(label="Output zip path", value=str(EXPORT_DIR / "command_classifier_export.zip"))
            export_btn = gr.Button("Export", variant="primary")
            export_result = gr.JSON(label="Export result")

            def on_export(ckpt, out_zip):
                import torch
                from command_classifier.config import (
                    AUDIO_DURATION_S,
                    CONFIDENCE_THRESHOLD,
                    F_MAX,
                    F_MIN,
                    HOP_LENGTH,
                    N_FFT,
                    N_MELS,
                    SAMPLE_RATE,
                )
                from command_classifier.export.package import create_export_zip
                from command_classifier.export.quantize import export_onnx, quantize_onnx
                from command_classifier.model.classifier import build_model

                payload = torch.load(ckpt, map_location="cpu")
                model = build_model(pretrained=False, num_classes=1)
                model.load_state_dict(payload["model_state"], strict=True)

                EXPORT_DIR.mkdir(parents=True, exist_ok=True)
                fp32_path = EXPORT_DIR / "model_fp32.onnx"
                int8_path = EXPORT_DIR / "model_int8.onnx"
                export_onnx(model, fp32_path)
                quantize_onnx(fp32_path, int8_path)

                config = {
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
                    "confidence_threshold": float(payload.get("metrics", {}).get("optimal_threshold", CONFIDENCE_THRESHOLD)),
                }
                zip_result = create_export_zip(
                    model_onnx_path=int8_path,
                    config=config,
                    export_dir=EXPORT_DIR,
                    zip_path=Path(out_zip),
                )
                return {"zip": str(zip_result), "fp32_onnx": str(fp32_path), "int8_onnx": str(int8_path)}

            export_btn.click(fn=on_export, inputs=[export_ckpt, export_zip], outputs=[export_result])

    return demo


def main(backbone_ckpt: Optional[str] = None) -> None:  # pragma: no cover
    demo = create_app(backbone_ckpt=backbone_ckpt)
    demo.launch()


if __name__ == "__main__":  # pragma: no cover
    main()
