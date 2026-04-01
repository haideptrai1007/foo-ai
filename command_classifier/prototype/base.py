from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class BasePrototype(ABC):
    """
    Prototype-based multi-command classifier.

    Workflow:
        proto.fit({"lights_on": [wav1, wav2, ...], "stop": [wav3, ...]})
        command, similarity, all_scores = proto.predict(new_wav, threshold=0.75)
        proto.save("prototypes.npz", method_name="logmel", threshold=0.75, audio_config={...})

    Classification is cosine similarity to the nearest class prototype.
    Each prototype is the L2-normalised mean of all clip embeddings for that command.
    """

    def __init__(self) -> None:
        self.prototypes: Dict[str, torch.Tensor] = {}
        self._speaker_mean: Optional[torch.Tensor] = None
        self._calibration_floor: float = 0.0

    @abstractmethod
    def extract_embedding(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract a 1-D embedding from a waveform.

        Args:
            waveform: (1, n_samples) float32 tensor at SAMPLE_RATE.

        Returns:
            (D,) float32 tensor — NOT normalised (normalisation happens in fit/predict).
        """
        ...

    def fit(self, command_waveforms: Dict[str, List[torch.Tensor]]) -> Dict[str, int]:
        """
        Compute one prototype per command.

        Applies prototype centering: subtracts the mean of all raw prototypes
        before L2-normalising. This removes the shared speaker-identity component
        so cosine similarity measures word content, not voice signature.

        Returns:
            {command_name: number_of_clips_used}
        """
        raw: Dict[str, torch.Tensor] = {}
        raw_clips: Dict[str, List[torch.Tensor]] = {}
        stats: Dict[str, int] = {}
        for cmd, waveforms in command_waveforms.items():
            if not waveforms:
                continue
            embeddings = [self.extract_embedding(w) for w in waveforms]
            raw_clips[cmd] = embeddings
            raw[cmd] = torch.stack(embeddings).mean(0)
            stats[cmd] = len(waveforms)

        if not raw:
            return stats

        # Speaker mean = centroid of all raw prototypes.
        # Subtracting it from each prototype (and from queries at inference)
        # removes the shared "this is this person's voice" bias.
        self._speaker_mean = torch.stack(list(raw.values())).mean(0)

        self.prototypes = {}
        for cmd, emb in raw.items():
            self.prototypes[cmd] = F.normalize(emb - self._speaker_mean, dim=0)

        # Calibration: compute each training clip's similarity to its own
        # prototype.  The global minimum becomes the rejection floor — any
        # query scoring below the worst real clip is likely an unknown word.
        in_class_sims: List[float] = []
        for cmd, clips in raw_clips.items():
            proto = self.prototypes[cmd]
            for emb in clips:
                centered = F.normalize(emb - self._speaker_mean, dim=0)
                in_class_sims.append(float((centered @ proto).item()))
        self._calibration_floor = min(in_class_sims) if in_class_sims else 0.0

        return stats

    # Reserved label for the rejection class.
    # Record 3-5 clips of random words / background noise with this name
    # to give the classifier an explicit "none of the above" prototype.
    OTHER_LABEL: str = "other"

    def predict(
        self,
        waveform: torch.Tensor,
        threshold: float = 0.0,
        margin: float = 0.05,
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Identify the closest command.

        Rejection has three layers:
          1. "other" class — if the user recorded clips named "other", the
             classifier uses them as an explicit rejection prototype. If "other"
             wins, return "none". This is the most reliable rejection method.
          2. calibration floor — the minimum in-class similarity observed
             during fit(). Acts as a data-driven threshold: any query scoring
             below the worst real training clip is rejected.
          3. threshold / margin fallbacks — raw cosine similarity gates.

        Args:
            threshold: reject if best similarity < this (after centering,
                       real commands score > 0; unknown words score near 0).
            margin:    reject if (best - second_best) < this.

        Returns:
            (best_command, best_similarity, {command: similarity, ...})
            best_command is "none" when rejected.
        """
        if not self.prototypes:
            raise RuntimeError("No prototypes — call fit() first.")

        raw_emb = self.extract_embedding(waveform)
        if self._speaker_mean is not None:
            raw_emb = raw_emb - self._speaker_mean
        query = F.normalize(raw_emb, dim=0)

        similarities = {
            cmd: float((query @ proto).item())
            for cmd, proto in self.prototypes.items()
        }

        best_cmd = max(similarities, key=similarities.__getitem__)
        best_sim = similarities[best_cmd]

        # "other" class wins → explicit rejection
        if best_cmd == self.OTHER_LABEL:
            return "none", best_sim, similarities

        # Calibration floor: reject if best score is below the worst
        # in-class similarity observed during fit().
        effective_threshold = max(threshold, self._calibration_floor)
        if best_sim < effective_threshold:
            return "none", best_sim, similarities

        if margin > 0.0 and len(similarities) >= 2:
            sorted_sims = sorted(similarities.values(), reverse=True)
            if sorted_sims[0] - sorted_sims[1] < margin:
                return "none", best_sim, similarities

        return best_cmd, best_sim, similarities

    def save(
        self,
        npz_path: Path,
        method_name: str,
        threshold: float,
        audio_config: Optional[Dict] = None,
    ) -> None:
        """
        Save prototypes to disk.

        Creates two files:
          - <npz_path>          — numpy arrays (no PyTorch needed at inference)
          - <npz_path>.json     — method, commands, threshold, audio config
        """
        npz_path = Path(npz_path)
        npz_path.parent.mkdir(parents=True, exist_ok=True)

        # Store arrays with stable numeric keys so command names (with spaces etc.)
        # never conflict with npz key rules.
        key_map: Dict[str, str] = {}  # "proto_0000" -> original command name
        arrays: Dict[str, np.ndarray] = {}
        for idx, (cmd, proto) in enumerate(self.prototypes.items()):
            k = f"proto_{idx:04d}"
            key_map[k] = cmd
            arrays[k] = proto.cpu().numpy()

        if self._speaker_mean is not None:
            arrays["_speaker_mean"] = self._speaker_mean.cpu().numpy()
        np.savez(str(npz_path), **arrays)

        meta = {
            "method": method_name,
            "threshold": threshold,
            "calibration_floor": self._calibration_floor,
            "embedding_dim": self.embedding_dim,
            "commands": key_map,
            "audio": audio_config or {},
        }
        json_path = npz_path.with_suffix(".json")
        json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def load(self, npz_path: Path) -> None:
        """Load prototypes saved with save()."""
        npz_path = Path(npz_path)
        data = np.load(str(npz_path))
        json_path = npz_path.with_suffix(".json")
        if json_path.exists():
            meta = json.loads(json_path.read_text(encoding="utf-8"))
            key_map: Dict[str, str] = meta.get("commands", {})
        else:
            key_map = {}
        self._speaker_mean = None
        self._calibration_floor = meta.get("calibration_floor", 0.0) if json_path.exists() else 0.0
        self.prototypes = {}
        for k in data.files:
            if k == "_speaker_mean":
                self._speaker_mean = torch.from_numpy(data[k])
                continue
            cmd = key_map.get(k, k)
            self.prototypes[cmd] = torch.from_numpy(data[k])

    @property
    def commands(self) -> List[str]:
        return list(self.prototypes.keys())

    @property
    def embedding_dim(self) -> int:
        if not self.prototypes:
            return 0
        return next(iter(self.prototypes.values())).shape[0]
