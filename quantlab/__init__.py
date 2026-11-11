"""QuantLab: int8 post-training quantization from scratch with measured error."""

from .analysis import compare_modes, imbalanced_matrix
from .integer import integer_audit
from .model import LinearClassifier, QuantizedClassifier, build_dataset, featurize
from .quantize import (
    dequantize, error_stats, quantize_per_channel, quantize_tensor, size_bytes,
)

__version__ = "0.3.0"

__all__ = [
    "integer_audit", "LinearClassifier", "QuantizedClassifier", "build_dataset", "compare_modes",
    "dequantize", "error_stats", "featurize", "imbalanced_matrix",
    "quantize_per_channel", "quantize_tensor", "size_bytes",
]
