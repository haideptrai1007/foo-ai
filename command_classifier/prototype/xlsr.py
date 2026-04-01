from __future__ import annotations

import torch

try:
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
except ImportError:
    Wav2Vec2Model = None

from command_classifier.config import SAMPLE_RATE
from command_classifier.prototype.base import BasePrototype

_MODEL_NAME = "facebook/wav2vec2-xls-r-300m"
_EXPECTED_SR = 16000
# XLS-R 300M has 24 transformer layers. Mid layers (8-12) capture phoneme
# content; later layers drift toward speaker/language identity.
_LAYER_IDX = 10


class XLSRPrototype(BasePrototype):
    """
    XLS-R 300M embedding prototype (1024-dim, 300M params, 128 languages).

    Uses facebook/wav2vec2-xls-r-300m as a frozen feature extractor.
    Embedding is mean-pooled hidden states from an intermediate transformer layer.

    Downloads ~1.2 GB on first run, cached by huggingface_hub.

    Pros:  128 languages, highest embedding quality, excellent phoneme discrimination.
    Cons:  Large model, slow on CPU, not practical for mobile deployment.
    """

    _model: object = None
    _feature_extractor: object = None

    def __init__(self) -> None:
        super().__init__()
        if Wav2Vec2Model is None:
            raise ImportError(
                "transformers is required for XLS-R embeddings. "
                "Install with: pip install transformers"
            )
        if SAMPLE_RATE != _EXPECTED_SR:
            raise ValueError(
                f"XLS-R expects {_EXPECTED_SR} Hz audio but "
                f"SAMPLE_RATE={SAMPLE_RATE}. Update config.py."
            )

    @classmethod
    def _get_model(cls):
        if cls._model is None:
            cls._feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(_MODEL_NAME)
            model = Wav2Vec2Model.from_pretrained(_MODEL_NAME)
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
            outputs = model(
                inputs.input_values,
                output_hidden_states=True,
            )
        # outputs.hidden_states: tuple of (1, T, 1024)
        hidden = outputs.hidden_states[min(_LAYER_IDX, len(outputs.hidden_states) - 1)]
        return hidden.squeeze(0).mean(0)  # (1024,)
