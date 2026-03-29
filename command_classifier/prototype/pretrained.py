from __future__ import annotations

import torch

try:
    import torchaudio
except ImportError:  # pragma: no cover
    torchaudio = None

from command_classifier.config import SAMPLE_RATE
from command_classifier.prototype.base import BasePrototype

_BUNDLE_NAME = "WAV2VEC2_BASE"
_EXPECTED_SR = 16000


class PretrainedEmbeddingPrototype(BasePrototype):
    """
    Pretrained wav2vec2-base embedding prototype.

    Uses WAV2VEC2_BASE (pretrained on 960h LibriSpeech) as a frozen feature
    extractor. Embedding is the mean-pooled last transformer layer → (768,).

    Key advantage over log-mel approaches: wav2vec2 has learned robust
    speech representations that are invariant to recording conditions,
    microphone gain, and moderate background noise.

    NOTE: The model (~360 MB) is downloaded once by torchaudio and cached.
    It is only needed during fit() — not at inference time, since prototypes
    are saved as plain numpy arrays in .npz.

    Pros:  Best noise robustness, highest discriminability for speech.
    Cons:  Slower to build prototypes (one model forward pass per clip).
           Requires torchaudio >= 0.10 with internet access on first run.
    """

    # Class-level model cache — shared across all instances to avoid
    # re-downloading / re-initialising on every fit() call.
    _model: object = None

    def __init__(self) -> None:
        super().__init__()
        if torchaudio is None:
            raise ImportError("torchaudio is required for pretrained embeddings.")
        if not hasattr(torchaudio, "pipelines"):
            raise ImportError(
                "torchaudio >= 0.10 is required for pipelines. "
                "Upgrade with: pip install --upgrade torchaudio"
            )
        if SAMPLE_RATE != _EXPECTED_SR:
            raise ValueError(
                f"{_BUNDLE_NAME} expects {_EXPECTED_SR} Hz audio but "
                f"SAMPLE_RATE={SAMPLE_RATE}. Update config.py."
            )

    @classmethod
    def _get_model(cls) -> object:
        """Lazy-load the wav2vec2 model on first call."""
        if cls._model is None:
            bundle = getattr(torchaudio.pipelines, _BUNDLE_NAME)
            model = bundle.get_model()
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            cls._model = model
        return cls._model

    def extract_embedding(self, waveform: torch.Tensor) -> torch.Tensor:
        model = self._get_model()
        with torch.no_grad():
            # extract_features returns (list_of_layer_outputs, lengths)
            features, _ = model.extract_features(waveform)
        # features[-1]: (1, T_frames, 768) — last transformer layer
        return features[-1].squeeze(0).mean(0)  # (768,)
