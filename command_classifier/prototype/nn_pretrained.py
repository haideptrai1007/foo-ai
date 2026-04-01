from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from command_classifier.prototype.base import BasePrototype
from command_classifier.prototype.pretrained import PretrainedEmbeddingPrototype


class NNPretrainedPrototype(BasePrototype):
    """
    Nearest-neighbor over wav2vec2 embeddings (768-dim).

    Combines the rich pretrained representations of wav2vec2-base with the
    nearest-neighbor strategy: keeps every clip separately and scores a
    query as the max cosine similarity across all clips for that command.

    This is the best-of-both-worlds approach: wav2vec2 features capture
    phoneme-level detail that log-mel mean/std cannot, and NN avoids the
    information loss from averaging with few samples.

    Pros:  Highest discriminability; robust to noise and recording conditions.
    Cons:  Requires torchaudio + wav2vec2 model (~360 MB download once);
           slower embedding extraction than log-mel approaches.
    """

    def __init__(self) -> None:
        super().__init__()
        self._pretrained = PretrainedEmbeddingPrototype()
        self._all_clips: Dict[str, List[torch.Tensor]] = {}

    def extract_embedding(self, waveform: torch.Tensor) -> torch.Tensor:
        return self._pretrained.extract_embedding(waveform)

    def fit(self, command_waveforms: Dict[str, List[torch.Tensor]]) -> Dict[str, int]:
        """Store L2-normalised wav2vec2 embeddings for every clip."""
        raw_all: Dict[str, List[torch.Tensor]] = {}
        stats: Dict[str, int] = {}

        for cmd, waveforms in command_waveforms.items():
            if not waveforms:
                continue
            embs = [self.extract_embedding(w) for w in waveforms]
            raw_all[cmd] = embs
            stats[cmd] = len(embs)

        if not raw_all:
            return stats

        all_embs = torch.stack([e for embs in raw_all.values() for e in embs])
        self._speaker_mean = all_embs.mean(0)

        self._all_clips = {}
        for cmd, embs in raw_all.items():
            centered = [F.normalize(e - self._speaker_mean, dim=0) for e in embs]
            self._all_clips[cmd] = centered

        self.prototypes = {}
        for cmd, embs in self._all_clips.items():
            self.prototypes[cmd] = torch.stack(embs).mean(0)

        # Calibration: leave-one-out max similarity per clip.
        in_class_sims: list[float] = []
        for cmd, clip_embs in self._all_clips.items():
            if len(clip_embs) < 2:
                continue
            for i, q in enumerate(clip_embs):
                others = [c for j, c in enumerate(clip_embs) if j != i]
                best = max(float((q @ c).item()) for c in others)
                in_class_sims.append(best)
        self._calibration_floor = min(in_class_sims) if in_class_sims else 0.0

        return stats

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
            sims = [float((query @ c).item()) for c in clip_embs]
            similarities[cmd] = max(sims)

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
