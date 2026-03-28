"""
Central configuration for the 1-class audio command classifier.

All modules should import constants from this file rather than hardcoding
hyperparameters or paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Tuple

# ----------------------------
# Base paths
# ----------------------------
BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent

# ----------------------------
# Audio
# ----------------------------
SAMPLE_RATE: Final[int] = 16000
AUDIO_DURATION_S: Final[float] = 2.0
AUDIO_SAMPLES: Final[int] = int(SAMPLE_RATE * AUDIO_DURATION_S)

# ----------------------------
# Mel-spectrogram
# ----------------------------
N_FFT: Final[int] = 256
HOP_LENGTH: Final[int] = 128
N_MELS: Final[int] = 40
F_MIN: Final[int] = 60
F_MAX: Final[int] = 7800
MEL_NORM: Final[str] = "slaney"

# ----------------------------
# Augmentation
# ----------------------------
AUG_FACTOR: Final[int] = 5
NOISE_SNR_RANGE: Final[Tuple[float, float]] = (5.0, 30.0)  # dB
PITCH_SHIFT_RANGE: Final[Tuple[float, float]] = (-2.0, 2.0)  # semitones
TIME_STRETCH_RANGE: Final[Tuple[float, float]] = (0.85, 1.15)

SPEC_FREQ_MASK_PARAM: Final[int] = 15
SPEC_TIME_MASK_PARAM: Final[int] = 20
SPEC_NUM_MASKS: Final[int] = 2

# ----------------------------
# Negative samples
# ----------------------------
NEG_RATIO: Final[float] = 1.0  # 1:1 pos:neg after augmentation
NEG_SOURCES: Final[Tuple[str, ...]] = (
    "silence",
    "white_noise",
    "pink_noise",
    "crowd_noise",
    "traffic_noise",
    "horn_noise",
)

# ----------------------------
# Model / optimization
# ----------------------------
NUM_CLASSES: Final[int] = 1  # single sigmoid output
DROPOUT: Final[float] = 0.3
BCRESNET_TAU: Final[int] = 1  # BCResNet scale: 1=smallest (edge), 2/3/6/8=larger

BATCH_SIZE: Final[int] = 64
NUM_EPOCHS: Final[int] = 40
LR: Final[float] = 1e-4
WEIGHT_DECAY: Final[float] = 1e-4
GRAD_CLIP_NORM: Final[float] = 1.0
NUM_WORKERS: Final[int] = 2

# ----------------------------
# Inference
# ----------------------------
CONFIDENCE_THRESHOLD: Final[float] = 0.5

# ----------------------------
# Export
# ----------------------------
ONNX_OPSET: Final[int] = 17
QUANTIZE_TYPE: Final[str] = "dynamic"  # "dynamic" or "static"

# ----------------------------
# Paths
# ----------------------------
DATA_DIR: Final[Path] = BASE_DIR / "data"
RAW_POSITIVE_DIR: Final[Path] = DATA_DIR / "positive"
RAW_NEGATIVE_DIR: Final[Path] = DATA_DIR / "negative"
AUGMENTED_DIR: Final[Path] = DATA_DIR / "augmented"
SPEECH_COMMANDS_SAMPLES_DIR: Final[Path] = DATA_DIR / "speech_commands_samples"
SPEECH_COMMANDS_SAMPLES_PER_CLASS: Final[int] = 5
CHECKPOINT_DIR: Final[Path] = Path("checkpoint")
EXPORT_DIR: Final[Path] = BASE_DIR / "export"

