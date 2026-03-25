from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch

from command_classifier.config import (
    AUDIO_DURATION_S,
    CONFIDENCE_THRESHOLD,
    F_MAX,
    F_MIN,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    ONNX_OPSET,
    SAMPLE_RATE,
    QUANTIZE_TYPE,
)
from command_classifier.export.package import create_export_zip
from command_classifier.export.quantize import export_onnx, quantize_onnx
from command_classifier.model.classifier import build_model


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Export checkpoint -> ONNX (FP32) -> INT8 quantized + zip bundle.")
    ap.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    ap.add_argument("--export-dir", default="./export", help="Directory for ONNX outputs and zip.")
    ap.add_argument("--opset", type=int, default=ONNX_OPSET, help="ONNX opset version.")
    ap.add_argument("--quantize-mode", default=QUANTIZE_TYPE, choices=["dynamic", "static"], help="Quantization mode.")
    ap.add_argument("--zip-path", default=None, help="Optional explicit output zip path.")
    args = ap.parse_args(argv)

    ckpt_path = Path(args.checkpoint)
    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(str(ckpt_path), map_location="cpu")

    model = build_model(pretrained=False, num_classes=1)
    model.load_state_dict(payload["model_state"], strict=True)

    fp32_path = export_dir / "model_fp32.onnx"
    int8_path = export_dir / "model_int8.onnx"

    export_onnx(model, fp32_path, opset=int(args.opset))
    quantize_onnx(fp32_path, int8_path, mode=str(args.quantize_mode))

    optimal_th = payload.get("metrics", {}).get("optimal_threshold", None)
    if optimal_th is None:
        optimal_th = CONFIDENCE_THRESHOLD

    config = {
        "sample_rate": int(SAMPLE_RATE),
        "audio_duration_s": float(AUDIO_DURATION_S),
        "n_fft": int(N_FFT),
        "hop_length": int(HOP_LENGTH),
        "n_mels": int(N_MELS),
        "f_min": int(F_MIN),
        "f_max": int(F_MAX),
        "image_size": 224,
        "imagenet_mean": [0.485, 0.456, 0.406],
        "imagenet_std": [0.229, 0.224, 0.225],
        "confidence_threshold": float(optimal_th),
    }

    zip_path = Path(args.zip_path) if args.zip_path is not None else (export_dir / "command_classifier_export.zip")
    result_zip = create_export_zip(
        model_onnx_path=int8_path,
        config=config,
        export_dir=export_dir,
        zip_path=zip_path,
    )

    print(json.dumps({"zip_path": str(result_zip), "fp32_onnx": str(fp32_path), "int8_onnx": str(int8_path)}, indent=2))


if __name__ == "__main__":
    main()

