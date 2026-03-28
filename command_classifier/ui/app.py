from __future__ import annotations

import queue
import tempfile
import threading
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


def _trim_silence(arr: np.ndarray, sr: int, top_db: float = 30.0) -> np.ndarray:
    """
    Trim leading/trailing silence using an energy threshold.

    Any frame whose RMS energy is below `top_db` dB relative to the peak
    frame is considered silent and stripped from the edges.

    Returns at least 0.1 s of audio so the clip is never empty.
    """
    if arr.ndim > 1:
        mono = arr.mean(axis=-1)
    else:
        mono = arr

    frame_len = max(1, int(sr * 0.02))  # 20 ms frames
    # Pad so we can reshape cleanly
    pad = (-len(mono)) % frame_len
    padded = np.concatenate([mono, np.zeros(pad, dtype=mono.dtype)])
    frames = padded.reshape(-1, frame_len)
    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))

    peak_rms = rms.max()
    if peak_rms == 0:
        return arr  # silent clip — leave unchanged

    threshold = peak_rms * (10.0 ** (-top_db / 20.0))
    active = np.where(rms >= threshold)[0]
    if len(active) == 0:
        return arr

    start_sample = int(active[0]) * frame_len
    end_sample = min(len(mono), int(active[-1] + 1) * frame_len)

    # Guarantee at least 0.1 s
    min_len = max(1, int(sr * 0.1))
    if end_sample - start_sample < min_len:
        center = (start_sample + end_sample) // 2
        start_sample = max(0, center - min_len // 2)
        end_sample = min(len(mono), start_sample + min_len)

    if arr.ndim > 1:
        return arr[start_sample:end_sample]
    return arr[start_sample:end_sample]


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
                btn_clear = gr.Button("Clear All", variant="stop")

            counter_display = gr.Textbox(
                label="Recorded so far",
                value="Recorded: 0 / 10",
                interactive=False,
            )

            def _counter_text(state, target):
                return f"Recorded: {len(state['positive'])} / {int(target)}"

            def save_positive(audio, state, target):
                if audio is None:
                    return state, _counter_text(state, target)
                if len(state["positive"]) >= int(target):
                    return state, _counter_text(state, target)
                sr, arr = audio
                arr = _trim_silence(arr.copy(), sr)
                state["positive"].append((sr, arr))
                return state, _counter_text(state, target)

            def clear_all(_state, target):
                new_state = {"positive": [], "negative": []}
                return new_state, _counter_text(new_state, target)

            btn_pos.click(fn=save_positive, inputs=[mic_audio, clips_state, n_samples], outputs=[clips_state, counter_display])
            btn_clear.click(fn=clear_all, inputs=[clips_state, n_samples], outputs=[clips_state, counter_display])

        # ── Tab 2: Train ─────────────────────────────────────────────────────
        with gr.Tab("2 · Train"):
            gr.Markdown("Train the few-shot classifier on your recorded samples.")

            with gr.Row():
                freeze_epochs_input = gr.Number(label="Freeze backbone epochs", value=15, precision=0)
                unfreeze_epochs_input = gr.Number(label="Unfreeze backbone epochs", value=5, precision=0)
                ckpt_dir_box = gr.Textbox(label="Checkpoint output dir", value=str(CHECKPOINT_DIR))

            if backbone_ckpt:
                gr.Markdown(f"**Backbone:** `{backbone_ckpt}`")
            else:
                gr.Markdown("**Backbone:** ImageNet MobileNetV3 (no Stage A checkpoint)")

            train_btn = gr.Button("Start Training", variant="primary")
            train_log = gr.Textbox(label="Training log", lines=12, interactive=False)
            train_result = gr.JSON(label="Final metrics")

            def on_train(state, freeze_epochs, unfreeze_epochs, ckpt_dir):
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

                    train_loader, val_loader, class_weights = create_dataloaders(
                        positive_dir=str(pos_dir),
                        negative_dir=str(neg_dir) if neg_dir else None,
                        seed=1234,
                    )

                    n_train = len(train_loader.dataset)
                    n_val = len(val_loader.dataset) if val_loader is not None else 0
                    val_info = (
                        f", {n_val} val samples | Train batches: {len(train_loader)}  Val batches: {len(val_loader)}"
                        if val_loader is not None
                        else " (no val split — all samples used for training)"
                    )
                    yield emit(
                        f"Dataset: {n_train} train samples{val_info}"
                    ), None
                    yield emit("Building model..."), None

                    model = build_model(pretrained=True, num_classes=1)
                    _load_backbone(model, backbone_ckpt)
                    model = prepare_model(model)

                    yield emit("Starting training..."), None

                    import command_classifier.training.trainer as _trainer_mod
                    _orig = _trainer_mod.NUM_EPOCHS
                    _trainer_mod.NUM_EPOCHS = int(freeze_epochs) + int(unfreeze_epochs)

                    log_q: queue.Queue = queue.Queue()
                    result_box: List[Any] = [None, None]  # [metrics, exception]

                    def _run_training():
                        try:
                            t = CNNTrainer(
                                model=model,
                                train_loader=train_loader,
                                val_loader=val_loader,
                                device=None,
                                checkpoint_dir=Path(ckpt_dir),
                                freeze_epochs=int(freeze_epochs),
                                pos_weight=class_weights["pos_weight"],
                            )
                            result_box[0] = t.train(log_fn=log_q.put)
                        except Exception as exc:
                            result_box[1] = exc
                        finally:
                            _trainer_mod.NUM_EPOCHS = _orig
                            log_q.put(None)  # sentinel

                    thread = threading.Thread(target=_run_training, daemon=True)
                    thread.start()

                    while True:
                        msg = log_q.get()
                        if msg is None:
                            break
                        yield emit(msg), None

                    thread.join()

                    if result_box[1] is not None:
                        yield emit(f"Error: {result_box[1]}"), None
                        return

                    metrics = result_box[0]

                yield emit("Training complete."), metrics

            train_click = train_btn.click(
                fn=on_train,
                inputs=[clips_state, freeze_epochs_input, unfreeze_epochs_input, ckpt_dir_box],
                outputs=[train_log, train_result],
            )

        # ── Tab 3: Test (real-time) ──────────────────────────────────────────
        with gr.Tab("3 · Test"):
            gr.Markdown("Record a clip and run inference. Inference time and probability are shown.")

            test_ckpt = gr.Textbox(label="Checkpoint (.pt) path", value=str(CHECKPOINT_DIR / "best.pt"))

            train_click.then(
                fn=lambda ckpt_dir: str(Path(ckpt_dir) / "best.pt"),
                inputs=[ckpt_dir_box],
                outputs=[test_ckpt],
            )
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
                if not Path(ckpt).exists():
                    return f"Checkpoint not found: {ckpt}", 0.0, 0.0
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
                    IMAGE_SIZE,
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
                    "image_size": IMAGE_SIZE,
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
