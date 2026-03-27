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
N_FFT: Final[int] = 512
HOP_LENGTH: Final[int] = 128
N_MELS: Final[int] = 64
F_MIN: Final[int] = 60
F_MAX: Final[int] = 7800
MEL_NORM: Final[str] = "slaney"

# ----------------------------
# Augmentation
# ----------------------------
AUG_FACTOR: Final[int] = 5
IMAGE_SIZE: Final[int] = 96  # input resolution for MobileNetV3 — smaller = faster on-device inference
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
    "random_speech",
    "crowd_noise",
    "traffic_noise",
    "horn_noise",
)

# ----------------------------
# Model / optimization
# ----------------------------
NUM_CLASSES: Final[int] = 1  # single sigmoid output
DROPOUT: Final[float] = 0.3
FREEZE_BACKBONE_EPOCHS: Final[int] = 5
UNFREEZE_LR_FACTOR: Final[float] = 0.1

BATCH_SIZE: Final[int] = 64
NUM_EPOCHS: Final[int] = 40
LR: Final[float] = 3e-4
WEIGHT_DECAY: Final[float] = 1e-4
LABEL_SMOOTHING: Final[float] = 0.05
POS_WEIGHT: Final[float] = 1.0
GRAD_CLIP_NORM: Final[float] = 1.0
EARLY_STOPPING_PATIENCE: Final[int] = 8
SCHEDULER: Final[str] = "OneCycleLR"
NUM_WORKERS: Final[int] = 2

# ----------------------------
# Validation
# ----------------------------
VAL_SPLIT: Final[float] = 0.15
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
CHECKPOINT_DIR: Final[Path] = Path("checkpoint")
EXPORT_DIR: Final[Path] = BASE_DIR / "export"

