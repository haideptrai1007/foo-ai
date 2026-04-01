from command_classifier.prototype.logmel import LogMelPrototype
from command_classifier.prototype.logmel_delta import LogMelDeltaPrototype
from command_classifier.prototype.logmel_meanstd import LogMelMeanStdPrototype
from command_classifier.prototype.nearest_neighbor import NearestNeighborPrototype
from command_classifier.prototype.nn_pretrained import NNPretrainedPrototype
from command_classifier.prototype.pretrained import PretrainedEmbeddingPrototype
from command_classifier.prototype.segmented_nearest_neighbor import SegmentedNearestNeighborPrototype
from command_classifier.prototype.weighted_nearest_neighbor import WeightedNearestNeighborPrototype
from command_classifier.prototype.whisper_tiny import WhisperTinyPrototype
from command_classifier.prototype.whisper_base import WhisperBasePrototype
from command_classifier.prototype.xlsr import XLSRPrototype

__all__ = [
    "LogMelPrototype",
    "LogMelDeltaPrototype",
    "LogMelMeanStdPrototype",
    "NearestNeighborPrototype",
    "NNPretrainedPrototype",
    "PretrainedEmbeddingPrototype",
    "SegmentedNearestNeighborPrototype",
    "WeightedNearestNeighborPrototype",
    "WhisperTinyPrototype",
    "WhisperBasePrototype",
    "XLSRPrototype",
]
