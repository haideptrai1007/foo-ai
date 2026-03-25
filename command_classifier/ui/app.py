from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Tuple

from command_classifier.config import (
    CHECKPOINT_DIR,
    EXPORT_DIR,
    RAW_NEGATIVE_DIR,
    RAW_POSITIVE_DIR,
)
from command_classifier.model.classifier import build_model, prepare_model
from command_classifier.preprocessing.dataset import create_dataloaders
from command_classifier.training.trainer import CNNTrainer


def _load_gradio():
    try:
        import gradio as gr

        return gr
    except Exception as e:  # pragma: no cover
        raise ImportError("gradio is required for the UI. Install gradio to use ui/app.py.") from e


def create_app():
    gr = _load_gradio()

    with gr.Blocks(title="1-Class Audio Command Classifier") as demo:
        gr.Markdown("# 1-Class Audio Command Classifier\nBinary: is this clip the target command?")

        with gr.Tab("Train"):
            gr.Markdown("Provide directories of positive (and optional negative) clips, then start training.")
            pos_dir = gr.Textbox(label="Positive directory", value=str(RAW_POSITIVE_DIR))
            neg_dir = gr.Textbox(label="Negative directory (optional)", value=str(RAW_NEGATIVE_DIR))
            cmd_label = gr.Textbox(label="Command label (for display only)", value="target_command")

            start_btn = gr.Button("Start Training")
            out_metrics = gr.JSON(label="Training result")

            def on_train(positive_dir: str, negative_directory: str, command_label: str) -> Any:
                # Build model and dataloaders
                train_loader, val_loader, _class_weights = create_dataloaders(
                    positive_dir=positive_dir,
                    negative_dir=negative_directory if negative_directory.strip() else None,
                    seed=1234,
                )
                model = build_model(pretrained=True, num_classes=1)
                model = prepare_model(model)
                trainer = CNNTrainer(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=None,
                    checkpoint_dir=CHECKPOINT_DIR,
                )
                metrics = trainer.train()
                return metrics

            start_btn.click(fn=on_train, inputs=[pos_dir, neg_dir, cmd_label], outputs=[out_metrics])

        with gr.Tab("Test"):
            gr.Markdown("Upload audio (wav/mp3/etc) or provide a local path and run inference.")
            checkpoint_path = gr.Textbox(label="Checkpoint (.pt) path", value=str((CHECKPOINT_DIR / "best_epoch_000.pt")))
            audio_path = gr.Textbox(label="Audio file path", value="")
            threshold = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="Threshold")
            run_btn = gr.Button("Run")

            out_is_cmd = gr.Textbox(label="Detected?", interactive=False)
            out_prob = gr.Number(label="Probability", interactive=False)

            def on_test(ckpt: str, audio: str, th: float) -> Tuple[str, float]:
                from command_classifier.inference.torch_infer import TorchInference

                infer = TorchInference(checkpoint_path=ckpt, threshold=th)
                is_cmd, prob = infer.predict(audio)
                return ("✅ COMMAND DETECTED" if is_cmd else "❌ NOT detected"), prob

            run_btn.click(fn=on_test, inputs=[checkpoint_path, audio_path, threshold], outputs=[out_is_cmd, out_prob])

        with gr.Tab("Export"):
            gr.Markdown("Export ONNX (FP32 + INT8) and package into a zip.")
            export_ckpt = gr.Textbox(label="Checkpoint (.pt) path", value=str((CHECKPOINT_DIR / "best_epoch_000.pt")))
            bundle_path = gr.Textbox(label="Output zip path", value=str((EXPORT_DIR / "command_classifier_export.zip")))
            export_btn = gr.Button("Export")
            out_export = gr.JSON(label="Export result")

            def on_export(ckpt: str, out_zip: str) -> Any:
                from command_classifier.export.quantize import export_onnx, quantize_onnx
                from command_classifier.export.package import create_export_zip
                from command_classifier.config import (
                    SAMPLE_RATE,
                    AUDIO_DURATION_S,
                    N_FFT,
                    HOP_LENGTH,
                    N_MELS,
                    F_MIN,
                    F_MAX,
                    CONFIDENCE_THRESHOLD,
                )

                ckpt_path = Path(ckpt)
                import torch
                payload = torch.load(str(ckpt_path), map_location="cpu")
                model = build_model(pretrained=False, num_classes=1)
                model.load_state_dict(payload["model_state"], strict=True)

                EXPORT_DIR.mkdir(parents=True, exist_ok=True)
                fp32_path = EXPORT_DIR / "model_fp32.onnx"
                int8_path = EXPORT_DIR / "model_int8.onnx"

                export_onnx(model, fp32_path)
                quantize_onnx(fp32_path, int8_path)

                optimal_th = CONFIDENCE_THRESHOLD
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
                    "confidence_threshold": float(optimal_th),
                }

                zip_result = create_export_zip(
                    model_onnx_path=int8_path,
                    config=config,
                    export_dir=EXPORT_DIR,
                    zip_path=Path(out_zip),
                )
                return {"zip_path": str(zip_result), "fp32_onnx": str(fp32_path), "int8_onnx": str(int8_path)}

            export_btn.click(fn=on_export, inputs=[export_ckpt, bundle_path], outputs=[out_export])

    return demo


def main() -> None:  # pragma: no cover
    demo = create_app()
    demo.launch()


if __name__ == "__main__":  # pragma: no cover
    main()

