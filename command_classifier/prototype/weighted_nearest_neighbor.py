from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from command_classifier.preprocessing.mel import create_mel_transform, waveform_to_mel
from command_classifier.prototype.nearest_neighbor import NearestNeighborPrototype


class WeightedNearestNeighborPrototype(NearestNeighborPrototype):
    """
    Weighted nearest-neighbor: top-k average instead of single max.

    Scores a query as the mean of the top-k cosine similarities across all
    clips for each command. More stable than raw max (less sensitive to one
    lucky clip) while still retaining the NN advantage over mean prototypes.

    Embedding: log-mel mean + std (80-dim), same as NearestNeighborPrototype.

    Pros:  More robust ranking than single-max NN; still no model needed.
    Cons:  Needs at least k clips per command to be meaningful (falls back
           to max when fewer clips exist).
    """

    def __init__(self, top_k: int = 3) -> None:
        super().__init__()
        self._top_k = top_k

    def predict(
        self,
        waveform: torch.Tensor,
        threshold: float = 0.0,
        margin: float = 0.05,
    ) -> Tuple[str, float, Dict[str, float]]:
        if not self._all_clips:
            raise RuntimeError("No clips — call fit() first.")

        raw_emb = self.extract_embedding(waveform)
        if self._speaker_mean is not None:
            raw_emb = raw_emb - self._speaker_mean
        query = F.normalize(raw_emb, dim=0)

        similarities: Dict[str, float] = {}
        for cmd, clip_embs in self._all_clips.items():
            sims = sorted(
                [float((query @ c).item()) for c in clip_embs], reverse=True
            )
            k = min(self._top_k, len(sims))
            similarities[cmd] = sum(sims[:k]) / k

        best_cmd = max(similarities, key=similarities.__getitem__)
        best_sim = similarities[best_cmd]

        if best_cmd == self.OTHER_LABEL:
            return "none", best_sim, similarities

        effective_threshold = max(threshold, self._calibration_floor)
        if best_sim < effective_threshold:
            return "none", best_sim, similarities

        if margin > 0.0 and len(similarities) >= 2:
            sorted_sims = sorted(similarities.values(), reverse=True)
            if sorted_sims[0] - sorted_sims[1] < margin:
                return "none", best_sim, similarities

        return best_cmd, best_sim, similarities
