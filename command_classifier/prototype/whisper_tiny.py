from __future__ import annotations

import torch

try:
    from transformers import WhisperModel, WhisperFeatureExtractor
except ImportError:
    WhisperModel = None

from command_classifier.config import SAMPLE_RATE
from command_classifier.prototype.base import BasePrototype

_MODEL_NAME = "openai/whisper-tiny"
_EXPECTED_SR = 16000
# Whisper encoder layers: 0-3 for tiny. Mid layers capture phoneme content
# better than the final layer which is tuned for decoder cross-attention.
_LAYER_IDX = 2


class WhisperTinyPrototype(BasePrototype):
    """
    Whisper-tiny encoder embedding prototype (384-dim, 39M params).

    Uses the encoder half of openai/whisper-tiny as a frozen feature extractor.
    Embedding is mean-pooled hidden states from an intermediate encoder layer.

    Supports 99+ languages. Downloads ~150 MB on first run, cached by
    huggingface_hub.

    Pros:  Multilingual, small, ONNX-exportable for mobile.
    Cons:  Slightly lower accuracy than larger Whisper variants.
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
        # waveform: (1, n_samples) → numpy 1-D
        audio_np = waveform.squeeze(0).numpy()
        inputs = fe(audio_np, sampling_rate=_EXPECTED_SR, return_tensors="pt")
        with torch.no_grad():
            outputs = model.encoder(
                inputs.input_features,
                output_hidden_states=True,
            )
        # outputs.hidden_states: tuple of (1, T, 384)
        hidden = outputs.hidden_states[min(_LAYER_IDX, len(outputs.hidden_states) - 1)]
        return hidden.squeeze(0).mean(0)  # (384,)
