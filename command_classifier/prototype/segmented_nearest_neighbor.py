from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from command_classifier.preprocessing.mel import create_mel_transform, waveform_to_mel
from command_classifier.prototype.nearest_neighbor import NearestNeighborPrototype


class SegmentedNearestNeighborPrototype(NearestNeighborPrototype):
    """
    Segmented nearest-neighbor: time-aware log-mel features.

    Splits the spectrogram into N temporal segments, computes mean+std per
    segment, and concatenates → (N_MELS * 2 * n_segments)-dim embedding.

    This captures temporal structure (e.g. beginning vs end of a word)
    that global mean+std loses. "stop" has energy at the start then silence;
    "go" is the reverse — segmented features distinguish them.

    Default 4 segments → 40 * 2 * 4 = 320-dim. Still lightweight, no model.

    Pros:  Captures temporal ordering; more discriminative than global stats.
    Cons:  Higher dimensionality; sensitive to alignment (clips must start
           similarly — energy gating / VAD helps).
    """

    def __init__(self, n_segments: int = 4) -> None:
        super().__init__()
        self._n_segments = n_segments

    def extract_embedding(self, waveform: torch.Tensor) -> torch.Tensor:
        mel = waveform_to_mel(waveform, self._mel_transform)  # (1, N_MELS, T)
        mel_2d = mel.squeeze(0)  # (N_MELS, T)
        T = mel_2d.shape[-1]

        parts = []
        seg_size = max(1, T // self._n_segments)
        for i in range(self._n_segments):
            start = i * seg_size
            end = start + seg_size if i < self._n_segments - 1 else T
            seg = mel_2d[:, start:end]
            parts.append(seg.mean(-1))
            parts.append(seg.std(-1))

        return torch.cat(parts, dim=0)
