"""STS-Net v0.2: clip-level sign language phonology model.

For the v0.1 per-frame BiLSTM model, see the standalone package in v0.1/.
"""

__version__ = "0.2.0"

from stsnet.clip_classifier import ClipClassifier
from stsnet.inference import ClipClassifierInference

__all__ = [
    "ClipClassifier",
    "ClipClassifierInference",
]
