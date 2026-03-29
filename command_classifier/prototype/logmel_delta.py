from __future__ import annotations

import torch

try:
    import torchaudio
except ImportError:  # pragma: no cover
    torchaudio = None

from command_classifier.preprocessing.mel import create_mel_transform, waveform_to_mel
from command_classifier.prototype.base import BasePrototype


class LogMelDeltaPrototype(BasePrototype):
    """
    Log-mel + delta + delta-delta prototype.

    Embedding: time-mean of [mel ; Δmel ; ΔΔmel] → (N_MELS * 3,) vector.

    Delta features capture temporal dynamics (how the spectrum changes over time),
    which improves discrimination between commands that share spectral content
    but differ in temporal shape (e.g. "on" vs "off" if they have similar vowels).

    Pros:  More noise-robust than raw MFCC stats; captures temporal motion.
    Cons:  3× the dimensions of log-mel only; still no learned features.
    """

    def __init__(self) -> None:
        super().__init__()
        if torchaudio is None:
            raise ImportError("torchaudio is required for delta features.")
        self._mel_transform = create_mel_transform()

    def extract_embedding(self, waveform: torch.Tensor) -> torch.Tensor:
        mel = waveform_to_mel(waveform, self._mel_transform)  # (1, N_MELS, T)
        mel_2d = mel.squeeze(0)                                # (N_MELS, T)
        delta = torchaudio.functional.compute_deltas(mel_2d)   # (N_MELS, T)
        delta2 = torchaudio.functional.compute_deltas(delta)   # (N_MELS, T)
        features = torch.cat([mel_2d, delta, delta2], dim=0)   # (N_MELS*3, T)
        return features.mean(-1)                                # (N_MELS*3,)
