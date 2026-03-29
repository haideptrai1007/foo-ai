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

        Returns:
            {command_name: number_of_clips_used}
        """
        self.prototypes = {}
        stats: Dict[str, int] = {}
        for cmd, waveforms in command_waveforms.items():
            if not waveforms:
                continue
            embeddings = [self.extract_embedding(w) for w in waveforms]
            mean_emb = torch.stack(embeddings).mean(0)
            self.prototypes[cmd] = F.normalize(mean_emb, dim=0)
            stats[cmd] = len(waveforms)
        return stats

    def predict(
        self,
        waveform: torch.Tensor,
        threshold: float = 0.0,
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Identify the closest command.

        Returns:
            (best_command, best_similarity, {command: similarity, ...})
            best_command is "none" when best_similarity < threshold.
        """
        if not self.prototypes:
            raise RuntimeError("No prototypes — call fit() first.")
        query = F.normalize(self.extract_embedding(waveform), dim=0)
        similarities = {
            cmd: float((query @ proto).item())
            for cmd, proto in self.prototypes.items()
        }
        best_cmd = max(similarities, key=similarities.__getitem__)
        best_sim = similarities[best_cmd]
        if best_sim < threshold:
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

        np.savez(str(npz_path), **arrays)

        meta = {
            "method": method_name,
            "threshold": threshold,
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
        self.prototypes = {}
        for k in data.files:
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
