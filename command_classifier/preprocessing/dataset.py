from __future__ import annotations

import math
import os
import random
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from command_classifier.config import (
    AUG_FACTOR,
    AUDIO_SAMPLES,
    BATCH_SIZE,
    NEG_RATIO,
    NEG_SOURCES,
    SPEECH_COMMANDS_SAMPLES_DIR,
    SPEECH_COMMANDS_SAMPLES_PER_CLASS,
)

from command_classifier.preprocessing.audio import load_audio
from command_classifier.preprocessing.augmentation import (
    AugmentationPipeline,
    generate_crowd_noise,
    generate_traffic_noise,
    generate_horn_noise,
)
from command_classifier.preprocessing.mel import create_mel_transform, waveform_to_mel


class CommandDataset(Dataset):
    """
    Lazy dataset: raw waveforms are loaded once, augmentation + mel conversion
    happen on-the-fly in __getitem__. This makes __init__ fast regardless of
    AUG_FACTOR, which is critical for few-shot training.

    Virtual length = num_raw_waveforms * AUG_FACTOR.
    Each index maps to (raw_waveform_idx, aug_slot) so every raw sample appears
    AUG_FACTOR times with a fresh random augmentation each time.
    """

    def __init__(
        self,
        positive_paths: Sequence[Path],
        negative_paths: Optional[Sequence[Path]] = None,
        augment: bool = True,
        seed: int = 1234,
    ) -> None:
        super().__init__()
        self.positive_paths = list(positive_paths)
        self.negative_paths = list(negative_paths) if negative_paths is not None else []
        self.augment = augment
        self.seed = seed

        self.pipeline = AugmentationPipeline()
        self.mel_transform = create_mel_transform().to("cpu")

        self._waveforms: List[torch.Tensor] = []
        self.labels: List[torch.Tensor] = []

        self._load_waveforms()

        # Dynamic aug factor: target 10–50 augmented samples per class.
        # Scales with how many originals you have; never exceeds AUG_FACTOR cap.
        if self.augment:
            n_pos = sum(1 for l in self.labels if float(l) > 0.5)
            n_neg = len(self.labels) - n_pos
            n_min = max(1, min(n_pos, n_neg))
            target = max(10, min(50, n_min * 3))
            self.aug_factor: int = min(AUG_FACTOR, max(1, math.ceil(target / n_min)))
        else:
            self.aug_factor = 1

        # Pre-compute augmented mel spectrograms in parallel — eliminates per-epoch cost.
        self._cache: List[torch.Tensor] = []
        self._build_cache()

    def _generate_negative_waveforms(
        self, positive_waveforms: Sequence[torch.Tensor], target_neg_base: int, seed: int
    ) -> List[torch.Tensor]:
        random.seed(seed)
        negs: List[torch.Tensor] = []
        while len(negs) < target_neg_base:
            src = random.choice(list(NEG_SOURCES))
            base_waveform = random.choice(list(positive_waveforms))
            if src == "silence":
                w = torch.zeros_like(base_waveform)
                # Slight noise so the mel isn't perfectly flat.
                w = w + 1e-4 * torch.randn_like(w)
            elif src == "white_noise":
                amp = random.uniform(0.05, 0.2)
                w = torch.randn_like(base_waveform) * amp
            elif src == "pink_noise":
                white = torch.randn_like(base_waveform)
                pink = torch.cumsum(white, dim=-1)
                pink_rms = pink.pow(2).mean().sqrt().clamp(min=1e-8)
                amp = random.uniform(0.05, 0.2)
                w = pink / pink_rms * amp
            elif src == "crowd_noise":
                w = generate_crowd_noise(AUDIO_SAMPLES)
            elif src == "traffic_noise":
                w = generate_traffic_noise(AUDIO_SAMPLES)
            elif src == "horn_noise":
                w = generate_horn_noise(AUDIO_SAMPLES)
            else:
                continue

            negs.append(w)
        return negs

    def _load_waveforms(self) -> None:
        """Load raw waveforms once. Fast — no augmentation or mel conversion here."""
        random.seed(self.seed)

        if len(self.positive_paths) == 0:
            raise ValueError("No positive samples found.")

        positive_waveforms: List[torch.Tensor] = []
        for p in self.positive_paths:
            try:
                positive_waveforms.append(load_audio(str(p)))
            except Exception as e:
                warnings.warn(f"Skipping positive '{p}': {e}")

        if len(positive_waveforms) == 0:
            raise ValueError("All positive samples failed to load.")

        target_neg_base = max(1, int(round(len(positive_waveforms) * float(NEG_RATIO))))
        negative_waveforms = self._generate_negative_waveforms(
            positive_waveforms=positive_waveforms,
            target_neg_base=target_neg_base,
            seed=self.seed + 1,
        )

        if len(self.negative_paths) > 0:
            for p in self.negative_paths:
                try:
                    negative_waveforms.append(load_audio(str(p)))
                except Exception as e:
                    warnings.warn(f"Skipping negative '{p}': {e}")

        # Load bundled SpeechCommands samples if available (capped per class to keep build fast)
        if SPEECH_COMMANDS_SAMPLES_DIR.exists():
            sc_paths = sorted(SPEECH_COMMANDS_SAMPLES_DIR.rglob("*.wav"))
            random.shuffle(sc_paths)
            sc_paths = sc_paths[:SPEECH_COMMANDS_SAMPLES_PER_CLASS * 35]
            for p in sc_paths:
                try:
                    negative_waveforms.append(load_audio(str(p)))
                except Exception as e:
                    warnings.warn(f"Skipping speech commands sample '{p}': {e}")

        for w in positive_waveforms:
            self._waveforms.append(w)
            self.labels.append(torch.tensor([1.0], dtype=torch.float32))

        for w in negative_waveforms:
            self._waveforms.append(w)
            self.labels.append(torch.tensor([0.0], dtype=torch.float32))

    def _build_cache(self) -> None:
        """Pre-compute augmented mel spectrograms in parallel and cache in RAM.

        Audio augmentation + FFT mel conversion happen once here across a thread pool.
        SpecAugment remains on-the-fly in __getitem__ — it's a fast tensor op.
        """
        augment = self.augment
        pipeline = self.pipeline
        mel_transform = self.mel_transform

        def _one(wf: torch.Tensor) -> torch.Tensor:
            wf_aug = pipeline.augment_audio(wf) if augment else wf
            return waveform_to_mel(wf_aug, mel_transform)

        tasks = [wf for wf in self._waveforms for _ in range(self.aug_factor)]
        workers = min(4, max(1, os.cpu_count() or 1))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            self._cache = list(ex.map(_one, tasks))

    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        mel_db = self._cache[idx]
        if self.augment:
            mel_db = self.pipeline.augment_spectrogram(mel_db)
        label = self.labels[idx // self.aug_factor]
        return mel_db, label  # (1, N_MELS, T_frames) — BCResNet-ready


def _audio_extensions() -> Tuple[str, ...]:
    return (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma")


def _list_audio_files(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    exts = _audio_extensions()
    paths: List[Path] = []
    for p in directory.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            paths.append(p)
    return sorted(paths)


def create_dataloaders(positive_dir: str, negative_dir: Optional[str] = None, seed: int = 1234):
    """
    Load audio, build augmented in-memory dataset, then split into train/val.

    Args:
        positive_dir: Directory containing positive audio clips.
        negative_dir: Optional directory containing negative audio clips.
        seed: Random seed for stratified split.

    Returns:
        (train_loader, val_loader, class_weights_dict)
    """

    pos_path = Path(positive_dir)
    neg_path = Path(negative_dir) if negative_dir is not None else None

    positive_paths = _list_audio_files(pos_path)
    negative_paths = _list_audio_files(neg_path) if neg_path is not None else []

    if len(positive_paths) == 0:
        raise ValueError(f"No positive audio files found in: {positive_dir}")

    dataset = CommandDataset(
        positive_paths=positive_paths,
        negative_paths=negative_paths if len(negative_paths) > 0 else None,
        augment=True,
        seed=seed,
    )

    # Use ALL samples for training — no val split.
    # With few-shot data every sample is precious; val metrics on 1-2 raw clips are meaningless.
    num_raw = len(dataset._waveforms)
    y_raw = torch.stack(dataset.labels).squeeze(1).numpy()  # shape: (num_raw,)
    num_aug = dataset.aug_factor

    all_idx = [r * num_aug + a for r in range(num_raw) for a in range(num_aug)]
    train_subset = torch.utils.data.Subset(dataset, all_idx)

    effective_batch = min(BATCH_SIZE, max(1, len(train_subset)))
    train_loader = DataLoader(
        train_subset,
        batch_size=effective_batch,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    # For BCEWithLogitsLoss, often `pos_weight = num_neg/num_pos`.
    num_pos = int((y_raw == 1).sum())
    num_neg = int((y_raw == 0).sum())
    pos_weight = float(num_neg) / float(max(1, num_pos))

    class_weights = {"pos_weight": pos_weight, "num_pos": num_pos, "num_neg": num_neg}
    return train_loader, None, class_weights

