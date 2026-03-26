from __future__ import annotations

import random
import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from command_classifier.config import (
    AUG_FACTOR,
    AUDIO_SAMPLES,
    IMAGE_SIZE,
    NEG_RATIO,
    NEG_SOURCES,
    NUM_WORKERS,
    VAL_SPLIT,
    BATCH_SIZE,
)

from command_classifier.preprocessing.audio import load_audio
from command_classifier.preprocessing.augmentation import AugmentationPipeline
from command_classifier.preprocessing.mel import create_mel_transform, mel_to_image, waveform_to_mel


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
        self.mel_transform = create_mel_transform()

        self._waveforms: List[torch.Tensor] = []
        self.labels: List[torch.Tensor] = []

        self._load_waveforms()

        # Pre-compute all augmented mel images once and cache in memory.
        # Cost: ~17 MB for 30 samples — negligible. Eliminates per-epoch mel recomputation.
        self._cache: List[torch.Tensor] = []  # length = num_raw * AUG_FACTOR
        self._build_cache()

    @staticmethod
    def _collect_paths(directory: Path, extensions: Iterable[str]) -> List[Path]:
        exts = {e.lower().lstrip(".") for e in extensions}
        paths: List[Path] = []
        if not directory.exists():
            return paths
        for p in directory.iterdir():
            if p.is_file() and p.suffix.lower().lstrip(".") in exts:
                paths.append(p)
        return sorted(paths)

    def _mutate_positive_for_negative(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Synthesize "wrong command" negatives from positive waveform.
        """
        # 3 mutation families, pick one randomly.
        choice = random.random()
        if choice < 0.33:
            # Heavy pitch shift: approximated by using smaller torchaudio pitch shift
            # but this requires torchaudio; we avoid that here by just time-reversing
            # as a robust fallback.
            return waveform.flip(-1)
        if choice < 0.66:
            # Chopped segments: concatenate shuffled segments
            n = waveform.size(-1)
            cut1 = int(n * random.uniform(0.2, 0.4))
            cut2 = int(n * random.uniform(0.6, 0.8))
            seg1 = waveform[:, :cut1]
            seg2 = waveform[:, cut1:cut2]
            seg3 = waveform[:, cut2:]
            segs = [seg1, seg2, seg3]
            random.shuffle(segs)
            out = torch.cat(segs, dim=-1)
        else:
            # Shuffled small windows
            n = waveform.size(-1)
            num_windows = random.randint(3, 6)
            window = n // num_windows
            chunks = []
            for i in range(num_windows):
                start = i * window
                end = (i + 1) * window if i < num_windows - 1 else n
                chunks.append(waveform[:, start:end])
            random.shuffle(chunks)
            out = torch.cat(chunks, dim=-1)

        # Pad/truncate to fixed length
        if out.size(-1) < AUDIO_SAMPLES:
            out = torch.nn.functional.pad(out, (0, AUDIO_SAMPLES - out.size(-1)))
        elif out.size(-1) > AUDIO_SAMPLES:
            out = out[:, :AUDIO_SAMPLES]
        return out

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
            elif src == "random_speech":
                w = self._mutate_positive_for_negative(base_waveform)
            else:
                w = self._mutate_positive_for_negative(base_waveform)

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

        for w in positive_waveforms:
            self._waveforms.append(w)
            self.labels.append(torch.tensor([1.0], dtype=torch.float32))

        for w in negative_waveforms:
            self._waveforms.append(w)
            self.labels.append(torch.tensor([0.0], dtype=torch.float32))

    def _build_cache(self) -> None:
        """Pre-compute augmented mel spectrograms and cache in RAM.

        Audio augmentation + FFT mel conversion (the expensive part) happens once here.
        SpecAugment + mel_to_image remain on-the-fly in __getitem__ — both are fast tensor ops.
        """
        num_aug = AUG_FACTOR if self.augment else 1
        for wf in self._waveforms:
            for _ in range(num_aug):
                wf_aug = self.pipeline.augment_audio(wf) if self.augment else wf
                self._cache.append(waveform_to_mel(wf_aug, self.mel_transform))

    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        mel_db = self._cache[idx]
        if self.augment:
            mel_db = self.pipeline.augment_spectrogram(mel_db)
        img = mel_to_image(mel_db, image_size=IMAGE_SIZE)
        label = self.labels[idx // (AUG_FACTOR if self.augment else 1)]
        return img, label


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
    num_aug = AUG_FACTOR

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

