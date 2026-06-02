"""STS-Net: multi-head sign language phonology model."""

__version__ = "0.2.0"

from stsnet.model import STSNet                           # v0.1 — per-frame BiLSTM
from stsnet.clip_classifier import ClipClassifier         # v0.2 — clip-level (default)
from stsnet.inference import STSNetInference, ClipClassifierInference

__all__ = [
    "STSNet",
    "ClipClassifier",
    "STSNetInference",
    "ClipClassifierInference",
]
