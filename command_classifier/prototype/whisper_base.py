from __future__ import annotations

import torch

try:
    from transformers import WhisperModel, WhisperFeatureExtractor
except ImportError:
    WhisperModel = None

from command_classifier.config import SAMPLE_RATE
from command_classifier.prototype.base import BasePrototype

_MODEL_NAME = "openai/whisper-base"
_EXPECTED_SR = 16000
# Whisper-base has 6 encoder layers. Layer 3-4 captures phoneme content well.
_LAYER_IDX = 3


class WhisperBasePrototype(BasePrototype):
    """
    Whisper-base encoder embedding prototype (512-dim, 74M params).

    Uses the encoder half of openai/whisper-base as a frozen feature extractor.
    Embedding is mean-pooled hidden states from an intermediate encoder layer.

    Supports 99+ languages. Downloads ~290 MB on first run, cached by
    huggingface_hub.

    Pros:  Multilingual, good accuracy, ONNX-exportable for mobile.
    Cons:  Larger than whisper-tiny; still much smaller than XLS-R.
    """

    _model: object = None
    _feature_extractor: object = None

    def __init__(self) -> None:
        super().__init__()
        if WhisperModel is None:
            raise ImportError(
                "transformers is required for Whisper embeddings. "
                "Install with: pip install transformers"
            )
        if SAMPLE_RATE != _EXPECTED_SR:
            raise ValueError(
                f"Whisper expects {_EXPECTED_SR} Hz audio but "
                f"SAMPLE_RATE={SAMPLE_RATE}. Update config.py."
            )

    @classmethod
    def _get_model(cls):
        if cls._model is None:
            cls._feature_extractor = WhisperFeatureExtractor.from_pretrained(_MODEL_NAME)
            model = WhisperModel.from_pretrained(_MODEL_NAME)
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            cls._model = model
        return cls._model, cls._feature_extractor

    def extract_embedding(self, waveform: torch.Tensor) -> torch.Tensor:
        model, fe = self._get_model()
        audio_np = waveform.squeeze(0).numpy()
        inputs = fe(audio_np, sampling_rate=_EXPECTED_SR, return_tensors="pt")
        with torch.no_grad():
            outputs = model.encoder(
                inputs.input_features,
                output_hidden_states=True,
            )
        # outputs.hidden_states: tuple of (1, T, 512)
        hidden = outputs.hidden_states[min(_LAYER_IDX, len(outputs.hidden_states) - 1)]
        return hidden.squeeze(0).mean(0)  # (512,)
