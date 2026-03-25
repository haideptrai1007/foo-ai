from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None  # type: ignore[assignment]

import torch

from command_classifier.config import CONFIDENCE_THRESHOLD
from command_classifier.preprocessing.audio import load_audio, load_from_gradio
from command_classifier.preprocessing.mel import full_pipeline


AudioInput = Union[str, Tuple[int, np.ndarray], Any]


class ONNXInference:
    """
    ONNX Runtime inference helper mirroring TorchInference.

    Notes:
      - Preprocessing is done in Python/torch/torchaudio to match the training mel pipeline.
      - The runtime does not require the PyTorch model.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        config_path: Optional[Union[str, Path]] = None,
        threshold: Optional[float] = None,
        providers: Optional[List[str]] = None,
    ) -> None:
        if ort is None:  # pragma: no cover
            raise ImportError("onnxruntime is required for ONNX inference.")

        self.model_path = Path(model_path)
        self.providers = providers or ["CPUExecutionProvider"]

        sess_options = ort.SessionOptions()
        self.session = ort.InferenceSession(str(self.model_path), sess_options=sess_options, providers=self.providers)

        # Load threshold from export config.json if provided.
        export_threshold = None
        if config_path is not None:
            cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
            export_threshold = cfg.get("confidence_threshold", None)

        self.threshold = float(threshold if threshold is not None else (export_threshold if export_threshold is not None else CONFIDENCE_THRESHOLD))

        self.input_name = self.session.get_inputs()[0].name

    def _waveform_from_input(self, audio_input: AudioInput) -> torch.Tensor:
        if isinstance(audio_input, str):
            return load_audio(audio_input)
        if isinstance(audio_input, tuple) and len(audio_input) == 2:
            return load_from_gradio(audio_input)
        raise TypeError(f"Unsupported audio_input type: {type(audio_input)}")

    def predict(self, audio_input: AudioInput) -> Tuple[bool, float]:
        waveform = self._waveform_from_input(audio_input)
        img = full_pipeline(waveform, image_size=224)  # (3,224,224), normalized for MobileNet
        inp = img.unsqueeze(0).cpu().numpy().astype(np.float32)  # (1,3,224,224)

        outputs = self.session.run(None, {self.input_name: inp})
        logits = outputs[0]
        # logits shape (batch,) or (batch,1)
        logits_1d = np.array(logits).reshape(-1)
        prob = 1.0 / (1.0 + np.exp(-logits_1d[0]))
        is_command = bool(prob >= self.threshold)
        return is_command, float(prob)

    def predict_batch(self, audio_inputs: List[AudioInput]) -> List[Tuple[bool, float]]:
        imgs = []
        for x in audio_inputs:
            waveform = self._waveform_from_input(x)
            img = full_pipeline(waveform, image_size=224)
            imgs.append(img)

        batch = torch.stack(imgs, dim=0).cpu().numpy().astype(np.float32)
        outputs = self.session.run(None, {self.input_name: batch})
        logits = outputs[0]
        logits_1d = np.array(logits).reshape(-1)
        probs = 1.0 / (1.0 + np.exp(-logits_1d))
        results: List[Tuple[bool, float]] = []
        for p in probs.tolist():
            results.append((bool(p >= self.threshold), float(p)))
        return results

