from __future__ import annotations

import torch

from command_classifier.preprocessing.mel import create_mel_transform, waveform_to_mel
from command_classifier.prototype.base import BasePrototype


class LogMelMeanStdPrototype(BasePrototype):
    """
    Log-mel mean + std prototype.

    Embedding: [mean_over_time ; std_over_time] of log-mel → (N_MELS * 2,) = 80-dim.

    Adding std captures how much the spectrum *varies* over time — a key cue that
    distinguishes short words with similar average spectra (e.g. "foo" vs "mama").
    "foo" has a fast consonant burst then a vowel → high std in certain bands.
    "mama" has repeated nasal+vowel pattern → different std profile.

    Pros:  2× more discriminative than mean-only; no heavy model; pure numpy at
           inference (just mel → mean/std → cosine). Ideal for edge devices.
    Cons:  Still no temporal ordering — "foo" and "oof" look the same.
    """

    def __init__(self) -> None:
        super().__init__()
        self._mel_transform = create_mel_transform()

    def extract_embedding(self, waveform: torch.Tensor) -> torch.Tensor:
        mel = waveform_to_mel(waveform, self._mel_transform)  # (1, N_MELS, T)
        mel_2d = mel.squeeze(0)                                # (N_MELS, T)
        mean = mel_2d.mean(-1)                                 # (N_MELS,)
        std = mel_2d.std(-1)                                   # (N_MELS,)
        return torch.cat([mean, std], dim=0)                   # (N_MELS * 2,)
