from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Tuple

import torch

try:
    import torchaudio
except ImportError:  # pragma: no cover
    torchaudio = None


from command_classifier.config import (
    AUDIO_SAMPLES,
    SAMPLE_RATE,
    NOISE_SNR_RANGE,
    PITCH_SHIFT_RANGE,
    SPEC_FREQ_MASK_PARAM,
    SPEC_NUM_MASKS,
    SPEC_TIME_MASK_PARAM,
    TIME_STRETCH_RANGE,
)

Tensor = torch.Tensor


def _probe_torchaudio_ops() -> Tuple[bool, bool]:
    """Probe once at import time — cheap dummy call to detect available/fast ops."""
    if torchaudio is None:
        return False, False
    _dummy = torch.zeros(1, 1600)
    has_pitch = False
    has_speed = False
    try:
        torchaudio.functional.pitch_shift(_dummy, sample_rate=16000, n_steps=1.0)
        has_pitch = True
    except Exception:
        pass
    try:
        torchaudio.functional.speed(_dummy, orig_freq=16000, factor=1.05)
        has_speed = True
    except Exception:
        pass
    return has_pitch, has_speed


_HAS_PITCH_SHIFT, _HAS_SPEED = _probe_torchaudio_ops()


def _require_torchaudio() -> None:
    if torchaudio is None:  # pragma: no cover
        raise ImportError("torchaudio is required for augmentation. Install torchaudio to use this module.")


@dataclass
class AugmentationPipeline:
    """
    Stochastic audio + spectrogram augmentation pipeline.

    Audio-domain augmentations are applied to the fixed-length waveform,
    then mel conversion happens later.

    SpecAugment (frequency + time masking) is always applied to mel-db.
    """

    # Probability range per augmentation op
    audio_op_p_min: float = 0.5
    audio_op_p_max: float = 0.7
    spec_num_masks: int = SPEC_NUM_MASKS

    def _snr_to_noise_scale(self, waveform: Tensor, snr_db: float) -> float:
        """
        Compute noise scale for a desired SNR in dB.
        """

        # For waveform: assume waveform is (1, n_samples). Use RMS power.
        signal_rms = waveform.pow(2).mean().sqrt().clamp(min=1e-8)
        snr_linear = 10.0 ** (snr_db / 20.0)
        noise_rms = signal_rms / snr_linear
        return float(noise_rms)

    def _add_gaussian_noise(self, waveform: Tensor) -> Tensor:
        snr_db = random.uniform(float(NOISE_SNR_RANGE[0]), float(NOISE_SNR_RANGE[1]))
        noise_scale = self._snr_to_noise_scale(waveform, snr_db)
        noise = torch.randn_like(waveform) * noise_scale
        return waveform + noise

    def _add_pink_noise(self, waveform: Tensor) -> Tensor:
        """
        Simple pink noise: cumulative sum of white noise (1/f-ish),
        then normalized to match desired SNR RMS.
        """

        snr_db = random.uniform(float(NOISE_SNR_RANGE[0]), float(NOISE_SNR_RANGE[1]))
        noise_scale = self._snr_to_noise_scale(waveform, snr_db)

        white = torch.randn_like(waveform)
        pink = torch.cumsum(white, dim=-1)
        # Normalize to unit RMS then scale to target RMS
        pink_rms = pink.pow(2).mean().sqrt().clamp(min=1e-8)
        pink = pink / pink_rms * noise_scale
        return waveform + pink

    def _time_shift(self, waveform: Tensor) -> Tensor:
        # Circular roll by +/- 10% of length
        n = waveform.size(-1)
        max_shift = int(0.1 * n)
        shift = random.randint(-max_shift, max_shift)
        if shift == 0:
            return waveform
        return torch.roll(waveform, shifts=shift, dims=-1)

    def _pitch_shift(self, waveform: Tensor) -> Tensor:
        steps = random.uniform(float(PITCH_SHIFT_RANGE[0]), float(PITCH_SHIFT_RANGE[1]))
        return torchaudio.functional.pitch_shift(waveform, sample_rate=SAMPLE_RATE, n_steps=steps)

    def _volume_perturb(self, waveform: Tensor) -> Tensor:
        gain = random.uniform(0.7, 1.3)
        return waveform * gain

    def _polarity_invert(self, waveform: Tensor) -> Tensor:
        return -waveform if random.random() < 0.5 else waveform

    def _time_stretch(self, waveform: Tensor) -> Tensor:
        rate = random.uniform(float(TIME_STRETCH_RANGE[0]), float(TIME_STRETCH_RANGE[1]))
        result = torchaudio.functional.speed(waveform, orig_freq=SAMPLE_RATE, factor=rate)
        stretched = result[0] if isinstance(result, (tuple, list)) else result
        if stretched.size(-1) == AUDIO_SAMPLES:
            return stretched
        if stretched.size(-1) < AUDIO_SAMPLES:
            return torch.nn.functional.pad(stretched, (0, AUDIO_SAMPLES - stretched.size(-1)))
        excess = stretched.size(-1) - AUDIO_SAMPLES
        start = excess // 2
        return stretched[:, start : start + AUDIO_SAMPLES]

    def _should_apply(self) -> bool:
        p = random.uniform(self.audio_op_p_min, self.audio_op_p_max)
        return random.random() < p

    def augment_audio(self, waveform: Tensor) -> Tensor:
        """
        Apply 2-4 random audio-domain augmentations stochastically.
        Slow torchaudio ops (pitch_shift, speed) are only included if probed
        as available and fast at import time.
        """

        ops = [
            self._add_gaussian_noise,
            self._add_pink_noise,
            self._time_shift,
            self._volume_perturb,
            self._polarity_invert,
        ]
        if _HAS_PITCH_SHIFT:
            ops.append(self._pitch_shift)
        if _HAS_SPEED:
            ops.append(self._time_stretch)

        # Choose a random subset of size 2-4 by independent probabilities,
        # with a hard fallback to ensure we always apply at least 2 ops.
        random.shuffle(ops)
        applied = 0
        for op in ops:
            if applied >= 4:
                break
            if self._should_apply():
                waveform = op(waveform)
                applied += 1

        # Ensure variety: if too few ops were selected, force-add from front
        if applied < 2:
            for op in ops:
                if applied >= 2:
                    break
                waveform = op(waveform)
                applied += 1

        # Final fixed length safety (pitch shift can change length)
        if waveform.size(-1) != AUDIO_SAMPLES:
            if waveform.size(-1) < AUDIO_SAMPLES:
                waveform = torch.nn.functional.pad(waveform, (0, AUDIO_SAMPLES - waveform.size(-1)))
            else:
                excess = waveform.size(-1) - AUDIO_SAMPLES
                start = excess // 2
                waveform = waveform[:, start : start + AUDIO_SAMPLES]

        return waveform

    def augment_spectrogram(self, mel_db: Tensor) -> Tensor:
        """
        Apply SpecAugment to mel-db tensor.

        Args:
            mel_db: Tensor shape (1, N_MELS, T)

        Returns:
            Augmented mel-db with same shape.
        """

        _require_torchaudio()
        if mel_db.dim() != 3 or mel_db.size(0) != 1:
            raise ValueError(f"Expected mel_db shape (1, N_MELS, T), got {tuple(mel_db.shape)}")

        # FrequencyMasking/TimeMasking accept (freq, time) or (batch, freq, time).
        mel = mel_db.clone()
        for _ in range(self.spec_num_masks):
            freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=SPEC_FREQ_MASK_PARAM)
            time_mask = torchaudio.transforms.TimeMasking(time_mask_param=SPEC_TIME_MASK_PARAM)
            mel = freq_mask(mel)
            mel = time_mask(mel)
        return mel

