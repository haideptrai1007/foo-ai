from __future__ import annotations

import io
import warnings
from functools import lru_cache
from typing import Any, Optional, Tuple, Union, cast

import torch

try:
    import torchaudio
except ImportError:  # pragma: no cover
    torchaudio = None

from command_classifier.config import AUDIO_SAMPLES, SAMPLE_RATE


Tensor = torch.Tensor


def _require_torchaudio() -> None:
    if torchaudio is None:  # type: ignore[truthy-function]
        raise ImportError("torchaudio is required for audio preprocessing. Install torchaudio to use this module.")


@lru_cache(maxsize=8)
def _get_resampler(orig_sr: int, target_sr: int, dtype: torch.dtype) -> "torchaudio.transforms.Resample":
    """
    Cache torchaudio Resample modules by (orig_sr, target_sr, dtype).

    Note: device placement is handled at call-time by moving the module
    to the waveform device.
    """

    _require_torchaudio()
    resampler = torchaudio.transforms.Resample(orig_sr=orig_sr, new_sr=target_sr)
    # Keep weights/buffers in the same dtype for fewer surprises.
    return cast("torchaudio.transforms.Resample", resampler.to(dtype=dtype))


def normalize_waveform(waveform: Tensor) -> Tensor:
    """
    Peak-normalize to [-1, 1]. Silent audio returns all-zeros.

    Args:
        waveform: Tensor of shape (1, n_samples) or (channels, n_samples)

    Returns:
        Normalized waveform with same shape/dtype/device.
    """

    max_abs = waveform.abs().max()
    if max_abs.numel() == 0:
        return waveform
    if float(max_abs) == 0.0:
        return torch.zeros_like(waveform)
    return waveform / max_abs.clamp(min=1e-12)


def pad_or_truncate(waveform: Tensor, target_len: int, mode: str = "center") -> Tensor:
    """
    Pad with zeros on the right, or crop to a fixed length.

    Args:
        waveform: Tensor of shape (1, n_samples)
        target_len: Desired fixed length in samples.
        mode: "center" for inference, "random_crop" for training.

    Returns:
        Tensor of shape (1, target_len)
    """

    if waveform.dim() != 2:
        raise ValueError(f"Expected waveform with shape (channels, time), got {tuple(waveform.shape)}")
    if waveform.size(0) != 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    cur_len = waveform.size(1)
    if cur_len == target_len:
        return waveform
    if cur_len < target_len:
        pad = target_len - cur_len
        return torch.nn.functional.pad(waveform, (0, pad))

    # Crop
    excess = cur_len - target_len
    if mode == "random_crop":
        start = int(torch.randint(0, excess + 1, (1,)).item())
    elif mode == "center":
        start = excess // 2
    else:
        raise ValueError(f"Unknown pad_or_truncate mode: {mode}")

    return waveform[:, start : start + target_len]


def _force_mono(waveform: Tensor) -> Tensor:
    if waveform.dim() != 2:
        raise ValueError(f"Expected 2D waveform (channels, time), got {tuple(waveform.shape)}")
    if waveform.size(0) == 1:
        return waveform
    return waveform.mean(dim=0, keepdim=True)


def load_audio(path: str, target_sr: int = SAMPLE_RATE) -> Tensor:
    """
    Load audio from disk, resample to target_sr, and return fixed-length mono waveform.

    Args:
        path: Path to an audio file.
        target_sr: Desired sample rate.

    Returns:
        Tensor of shape (1, AUDIO_SAMPLES), float32.
    """

    _require_torchaudio()
    try:
        waveform, sr = torchaudio.load(path)  # (channels, time)
    except Exception as e:  # pragma: no cover
        warnings.warn(f"Failed to load audio '{path}': {e}")
        raise

    waveform = _force_mono(waveform)
    waveform = waveform.to(torch.float32)

    if sr != target_sr:
        resampler = _get_resampler(sr, target_sr, dtype=waveform.dtype).to(device=waveform.device)
        waveform = resampler(waveform)

    waveform = pad_or_truncate(waveform, AUDIO_SAMPLES, mode="center")
    waveform = normalize_waveform(waveform)
    return waveform


def load_from_bytes(audio_bytes: bytes, sr: int) -> Tensor:
    """
    Load audio from raw bytes (used by some UI flows).

    Args:
        audio_bytes: Bytes of the audio file/stream.
        sr: Source sample rate (best effort; if incorrect, mel will be off).

    Returns:
        Tensor of shape (1, AUDIO_SAMPLES)
    """

    _require_torchaudio()
    with io.BytesIO(audio_bytes) as bio:
        waveform, file_sr = torchaudio.load(bio)

    waveform = _force_mono(waveform).to(torch.float32)
    src_sr = int(file_sr) if file_sr is not None else int(sr)
    if src_sr != SAMPLE_RATE:
        resampler = _get_resampler(src_sr, SAMPLE_RATE, dtype=waveform.dtype).to(device=waveform.device)
        waveform = resampler(waveform)

    waveform = pad_or_truncate(waveform, AUDIO_SAMPLES, mode="center")
    waveform = normalize_waveform(waveform)
    return waveform


def load_from_gradio(audio: Any, sr: Optional[int] = None) -> Tensor:
    """
    Convert a Gradio `Audio` component output into a normalized mono waveform.

    Common Gradio formats:
      - (sample_rate, numpy_array)
      - numpy_array only (then `sr` must be provided)

    Args:
        audio: Gradio audio payload.
        sr: Sample rate if `audio` does not include it.

    Returns:
        Tensor of shape (1, AUDIO_SAMPLES).
    """

    _require_torchaudio()
    if audio is None:
        raise ValueError("No audio provided (None)")

    if isinstance(audio, tuple) and len(audio) == 2:
        in_sr = int(audio[0])
        arr = audio[1]
    else:
        if sr is None:
            raise ValueError("Gradio audio is missing sample_rate; pass `sr`.")
        in_sr = int(sr)
        arr = audio

    if not isinstance(arr, (list, tuple)) and not hasattr(arr, "shape"):
        raise TypeError(f"Unsupported audio array type: {type(arr)}")

    x = torch.as_tensor(arr, dtype=torch.float32)
    if x.dim() == 1:
        x = x.unsqueeze(0)  # (1, time)
    elif x.dim() == 2:
        # Could be (time, channels) or (channels, time). Heuristic:
        # treat the longer dimension as time.
        if x.size(0) < x.size(1):
            # (channels, time)
            pass
        else:
            # (time, channels) -> transpose
            x = x.transpose(0, 1)
    else:
        raise ValueError(f"Unexpected audio array dims: {tuple(x.shape)}")

    x = _force_mono(x)
    if in_sr != SAMPLE_RATE:
        resampler = _get_resampler(in_sr, SAMPLE_RATE, dtype=x.dtype).to(device=x.device)
        x = resampler(x)

    x = pad_or_truncate(x, AUDIO_SAMPLES, mode="center")
    x = normalize_waveform(x)
    return x

