from __future__ import annotations

from functools import lru_cache

import torch

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



