from __future__ import annotations

import torch

from command_classifier.preprocessing.mel import create_mel_transform, waveform_to_mel
from command_classifier.prototype.base import BasePrototype


class LogMelPrototype(BasePrototype):
    """
    Log-mel spectrogram prototype.

    Embedding: mean of log-mel over time axis → (N_MELS,) vector.

    Pros:  Fast, simple, robust to amplitude shifts (log compression).
    Cons:  Loses temporal dynamics — commands with same spectral shape
           but different onset/offset timing look identical.
    """

    def __init__(self) -> None:
        super().__init__()
        self._mel_transform = create_mel_transform()

    def extract_embedding(self, waveform: torch.Tensor) -> torch.Tensor:
        mel = waveform_to_mel(waveform, self._mel_transform)  # (1, N_MELS, T)
        return mel.squeeze(0).mean(-1)  # (N_MELS,)
