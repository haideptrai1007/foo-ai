from __future__ import annotations

from functools import lru_cache
from typing import Final

import torch
import torch.nn.functional as F

try:
    import torchaudio
except ImportError:  # pragma: no cover
    torchaudio = None

from command_classifier.config import (
    F_MAX,
    F_MIN,
    HOP_LENGTH,
    MEL_NORM,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
)


IMAGENET_MEAN: Final[list[float]] = [0.485, 0.456, 0.406]
IMAGENET_STD: Final[list[float]] = [0.229, 0.224, 0.225]


@lru_cache(maxsize=2)
def create_mel_transform() -> "torchaudio.transforms.MelSpectrogram":
    """
    Create and cache a MelSpectrogram transform.

    Returns:
        MelSpectrogram transform which outputs power spectrograms.
    """

    if torchaudio is None:  # pragma: no cover
        raise ImportError("torchaudio is required for mel-spectrogram preprocessing. Install torchaudio to use this module.")
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        f_min=F_MIN,
        f_max=F_MAX,
        power=2.0,
        norm=MEL_NORM,
    )


def waveform_to_mel(waveform: torch.Tensor, mel_transform: "torchaudio.transforms.MelSpectrogram") -> torch.Tensor:
    """
    Convert waveform to mel spectrogram in dB.

    Args:
        waveform: Tensor of shape (1, n_samples)
        mel_transform: torchaudio MelSpectrogram instance

    Returns:
        Mel dB tensor with shape (1, N_MELS, time_frames)
    """
    if torchaudio is None:  # pragma: no cover
        raise ImportError("torchaudio is required for mel-spectrogram preprocessing. Install torchaudio to use this module.")

    # MelSpectrogram outputs (channels, n_mels, time)
    mel_power = mel_transform(waveform)
    amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)
    mel_db = amp_to_db(mel_power)
    return mel_db


def mel_to_image(mel_db: torch.Tensor, image_size: int = 224) -> torch.Tensor:
    """
    Convert mel-dB tensor to an ImageNet-normalized 3x224x224 tensor.

    Args:
        mel_db: Tensor of shape (1, N_MELS, T)
        image_size: Output height/width (square)

    Returns:
        Tensor of shape (3, image_size, image_size)
    """

    if mel_db.dim() != 3 or mel_db.size(0) != 1:
        raise ValueError(f"Expected mel_db shape (1, N_MELS, T), got {tuple(mel_db.shape)}")

    # Per-sample min-max normalization to [0, 1]
    min_val = mel_db.amin(dim=(1, 2), keepdim=True)
    max_val = mel_db.amax(dim=(1, 2), keepdim=True)
    mel_01 = (mel_db - min_val) / (max_val - min_val).clamp(min=1e-6)

    # Resize: treat mel as 1-channel image (N=1, C=1, H=128, W=T)
    img_1c = F.interpolate(
        mel_01.unsqueeze(0),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)  # (1, 224, 224)

    img_3c = img_1c.repeat(3, 1, 1)  # (3, 224, 224)

    mean = torch.tensor(IMAGENET_MEAN, device=img_3c.device, dtype=img_3c.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=img_3c.device, dtype=img_3c.dtype).view(3, 1, 1)
    img_norm = (img_3c - mean) / std
    return img_norm


def full_pipeline(waveform: torch.Tensor, image_size: int = 224) -> torch.Tensor:
    """
    Convenience wrapper: waveform -> mel-db -> image tensor.
    """

    mel_transform = create_mel_transform().to(device=waveform.device, dtype=waveform.dtype)
    mel_db = waveform_to_mel(waveform, mel_transform)
    return mel_to_image(mel_db, image_size=image_size)

