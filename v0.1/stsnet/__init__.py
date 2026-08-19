"""STS-Net v0.1: per-frame multi-head sign language phonology model."""

__version__ = "0.1.0"

from stsnet.model import STSNet
from stsnet.inference import STSNetInference

__all__ = [
    "STSNet",
    "STSNetInference",
]
