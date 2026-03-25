from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from command_classifier.config import CONFIDENCE_THRESHOLD
from command_classifier.model.classifier import build_model, _unwrap
from command_classifier.preprocessing.audio import load_audio, load_from_bytes, load_from_gradio
from command_classifier.preprocessing.mel import create_mel_transform, mel_to_image, waveform_to_mel


AudioInput = Union[str, Tuple[int, np.ndarray], Tuple[np.ndarray, int], bytes, Any]


class TorchInference:
    """
    PyTorch inference helper for a single clip or a batch.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: Optional[torch.device] = None,
        threshold: Optional[float] = None,
        pretrained: bool = False,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.threshold = float(threshold if threshold is not None else CONFIDENCE_THRESHOLD)

        self.model = build_model(pretrained=pretrained, num_classes=1)
        payload = torch.load(str(self.checkpoint_path), map_location="cpu")
        state = payload.get("model_state", payload)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.mel_transform = create_mel_transform()
        self.mel_transform = self.mel_transform.to(device=self.device)

    def _waveform_from_input(self, audio_input: AudioInput) -> torch.Tensor:
        if isinstance(audio_input, str):
            return load_audio(audio_input)
        if isinstance(audio_input, bytes):
            # No sample rate provided; bytes should ideally include correct container.
            # If sample rate is needed externally, prefer passing a filepath or a (sr, np_array) tuple.
            raise ValueError("Bytes input is not supported without sample rate. Provide a file path or (sr, np.ndarray).")
        if isinstance(audio_input, tuple) and len(audio_input) == 2:
            # Accept (sr, np_array)
            if isinstance(audio_input[0], (int, np.integer)):
                return load_from_gradio(audio_input)
            # Accept (np_array, sr)
            return load_from_gradio((int(audio_input[1]), audio_input[0]))
        raise TypeError(f"Unsupported audio_input type: {type(audio_input)}")

    @torch.no_grad()
    def predict(self, audio_input: AudioInput) -> Tuple[bool, float]:
        waveform = self._waveform_from_input(audio_input)  # (1, n)
        waveform = waveform.to(self.device)

        mel_db = waveform_to_mel(waveform, self.mel_transform)
        img = mel_to_image(mel_db, image_size=224)  # (3, 224, 224)
        img = img.unsqueeze(0)  # (1, 3, 224, 224)

        logits = self.model(img).view(-1)
        prob = torch.sigmoid(logits)[0].item()
        is_command = prob >= self.threshold
        return is_command, float(prob)

    @torch.no_grad()
    def predict_batch(self, audio_inputs: List[AudioInput]) -> List[Tuple[bool, float]]:
        return [self.predict(x) for x in audio_inputs]

    def set_threshold(self, threshold: float) -> None:
        self.threshold = float(threshold)

