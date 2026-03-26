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
    In-memory dataset of mel-images + binary labels.

    Each underlying waveform (positive or negative) is augmented `AUG_FACTOR` times
    (unless augment=False) and converted into a 3x224x224 image tensor.
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

        self.images: List[torch.Tensor] = []
        self.labels: List[torch.Tensor] = []

        self._build_in_memory()

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

    def _waveform_to_image(self, waveform: torch.Tensor, label: float) -> Tuple[torch.Tensor, torch.Tensor]:
        mel_db = waveform_to_mel(waveform, self.mel_transform)
        if self.augment:
            mel_db = self.pipeline.augment_spectrogram(mel_db)
        img = mel_to_image(mel_db, image_size=224)
        return img, torch.tensor([label], dtype=torch.float32)

    def _build_in_memory(self) -> None:
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        if len(self.positive_paths) == 0:
            raise ValueError("No positive samples found.")

        # Load positives
        positive_waveforms: List[torch.Tensor] = []
        for p in self.positive_paths:
            try:
                positive_waveforms.append(load_audio(str(p)))
            except Exception as e:
                warnings.warn(f"Skipping positive '{p}': {e}")

        if len(positive_waveforms) == 0:
            raise ValueError("All positive samples failed to load.")

        # Generate a base set of negatives to match NEG_RATIO before augmentation.
        target_neg_base = max(1, int(round(len(positive_waveforms) * float(NEG_RATIO))))
        negative_waveforms = self._generate_negative_waveforms(
            positive_waveforms=positive_waveforms,
            target_neg_base=target_neg_base,
            seed=self.seed + 1,
        )

        # Optionally also load user-provided negatives and mix in.
        if len(self.negative_paths) > 0:
            for p in self.negative_paths:
                try:
                    negative_waveforms.append(load_audio(str(p)))
                except Exception as e:
                    warnings.warn(f"Skipping negative '{p}': {e}")

        num_aug = AUG_FACTOR if self.augment else 1

        # Convert all waveform variants to mel images
        for w in positive_waveforms:
            for _ in range(num_aug):
                wf = w
                if self.augment:
                    wf = self.pipeline.augment_audio(wf)
                img, y = self._waveform_to_image(wf, label=1.0)
                self.images.append(img)
                self.labels.append(y)

        for w in negative_waveforms:
            for _ in range(num_aug):
                wf = w
                if self.augment:
                    wf = self.pipeline.augment_audio(wf)
                img, y = self._waveform_to_image(wf, label=0.0)
                self.images.append(img)
                self.labels.append(y)

        # Sanity: keep everything aligned.
        if len(self.images) != len(self.labels):
            raise RuntimeError("Internal dataset build error: images/labels length mismatch.")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.labels[idx]


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

    # Stratified split based on label (sklearn optional)
    y = torch.stack(dataset.labels).squeeze(1).numpy()
    indices = list(range(len(dataset)))

    try:
        from sklearn.model_selection import train_test_split

        train_idx, val_idx = train_test_split(
            indices,
            test_size=float(VAL_SPLIT),
            random_state=seed,
            stratify=y,
        )
    except Exception:
        # Fallback stratified split without sklearn.
        rng = random.Random(seed)
        pos_idx = [i for i in indices if int(y[i]) == 1]
        neg_idx = [i for i in indices if int(y[i]) == 0]
        rng.shuffle(pos_idx)
        rng.shuffle(neg_idx)
        val_pos = max(1, int(round(len(pos_idx) * float(VAL_SPLIT))))
        val_neg = max(1, int(round(len(neg_idx) * float(VAL_SPLIT))))
        val_idx = pos_idx[:val_pos] + neg_idx[:val_neg]
        train_idx = [i for i in indices if i not in set(val_idx)]

    # Guard: ensure both splits have at least 2 samples.
    if len(val_idx) < 2 or len(train_idx) < 2:
        mid = len(indices) // 2
        train_idx = indices[:mid]
        val_idx = indices[mid:]

    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)

    # Few-shot friendly settings: no sample dropping, no multiprocessing overhead.
    effective_batch = min(BATCH_SIZE, max(1, len(train_subset)))
    train_loader = DataLoader(
        train_subset,
        batch_size=effective_batch,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=min(BATCH_SIZE, max(1, len(val_subset))),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    # For BCEWithLogitsLoss, often `pos_weight = num_neg/num_pos`.
    num_pos = int((y == 1).sum())
    num_neg = int((y == 0).sum())
    pos_weight = float(num_neg) / float(max(1, num_pos))

    class_weights = {"pos_weight": pos_weight, "num_pos": num_pos, "num_neg": num_neg}
    return train_loader, val_loader, class_weights

