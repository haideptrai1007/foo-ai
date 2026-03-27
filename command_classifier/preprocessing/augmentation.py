from __future__ import annotations

import math
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


def generate_crowd_noise(n_samples: int, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """Band-limited pink noise with slow AM — simulates crowd murmur."""
    white = torch.randn(1, n_samples)
    pink = torch.cumsum(white, dim=-1)
    # Band-pass to speech range (300–3000 Hz) via FFT brick-wall
    fft = torch.fft.rfft(pink, dim=-1)
    freqs = torch.fft.rfftfreq(n_samples, d=1.0 / sample_rate)
    fft = fft * ((freqs >= 300) & (freqs <= 3000)).float()
    noise = torch.fft.irfft(fft, n=n_samples, dim=-1)
    # Slow amplitude modulation (1–3 Hz) for crowd ebb/flow
    t = torch.linspace(0, n_samples / sample_rate, n_samples)
    mod = 0.7 + 0.3 * torch.sin(2 * math.pi * random.uniform(1.0, 3.0) * t)
    noise = noise * mod.unsqueeze(0)
    rms = noise.pow(2).mean().sqrt().clamp(min=1e-8)
    return noise / rms * random.uniform(0.05, 0.3)


def generate_traffic_noise(n_samples: int, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """Low-freq filtered noise + engine harmonics — simulates vehicle rumble."""
    white = torch.randn(1, n_samples)
    fft = torch.fft.rfft(white, dim=-1)
    freqs = torch.fft.rfftfreq(n_samples, d=1.0 / sample_rate)
    fft = fft * (freqs <= 250).float()
    noise = torch.fft.irfft(fft, n=n_samples, dim=-1)
    # Periodic engine harmonics
    t = torch.linspace(0, n_samples / sample_rate, n_samples)
    fundamental = random.uniform(60.0, 120.0)
    harmonics = sum((1.0 / k) * torch.sin(2 * math.pi * fundamental * k * t) for k in range(1, 5))
    combined = noise + 0.3 * harmonics.unsqueeze(0)
    rms = combined.pow(2).mean().sqrt().clamp(min=1e-8)
    return combined / rms * random.uniform(0.05, 0.25)


def generate_horn_noise(n_samples: int, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """Short tonal burst(s) at ~400–500 Hz — simulates a car horn."""
    t = torch.linspace(0, n_samples / sample_rate, n_samples)
    fundamental = random.uniform(380.0, 520.0)
    tone = torch.sin(2 * math.pi * fundamental * t) + 0.5 * torch.sin(2 * math.pi * 2 * fundamental * t)
    tone = tone.unsqueeze(0)
    # Envelope: 1–2 short honks
    envelope = torch.zeros(n_samples)
    total_s = n_samples / sample_rate
    for _ in range(random.randint(1, 2)):
        start_s = random.uniform(0.0, total_s * 0.5)
        dur_s = random.uniform(0.1, 0.4)
        s = int(start_s * sample_rate)
        e = min(n_samples, int((start_s + dur_s) * sample_rate))
        envelope[s:e] = 1.0
    tone = tone * envelope.unsqueeze(0) + 0.05 * torch.randn_like(tone)
    rms = tone.pow(2).mean().sqrt().clamp(min=1e-8)
    return tone / rms * random.uniform(0.1, 0.4)


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

    def _add_ambient_noise(self, waveform: Tensor) -> Tensor:
        """Mix one of crowd / traffic / horn noise into the waveform at a random SNR."""
        gen = random.choice([generate_crowd_noise, generate_traffic_noise, generate_horn_noise])
        noise = gen(waveform.size(-1)).to(waveform.device)
        snr_db = random.uniform(float(NOISE_SNR_RANGE[0]), float(NOISE_SNR_RANGE[1]))
        noise_scale = self._snr_to_noise_scale(waveform, snr_db)
        noise_rms = noise.pow(2).mean().sqrt().clamp(min=1e-8)
        return waveform + noise / noise_rms * noise_scale

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
            self._add_ambient_noise,
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

