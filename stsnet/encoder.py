"""
Pose stream encoder components for STS-Net.

TemporalConvBlock: single Conv1d layer with residual connection.
FrameEncoder: encodes one pose stream (B, T, J, 3) -> (B, T, D).
"""

import torch
import torch.nn as nn
from torch import Tensor


class TemporalConvBlock(nn.Module):
    """Single Conv1d layer with residual connection, LayerNorm, ReLU, Dropout."""

    def __init__(self, channels: int, kernel_size: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.norm    = nn.LayerNorm(channels)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, D, T)
        out = self.conv(x) + x
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        return self.dropout(self.relu(out))


class FrameEncoder(nn.Module):
    """
    Encodes one pose stream (B, T, J, 3) -> (B, T, D).

    Invalid frames (wrist joint NaN) are zeroed so they don't corrupt
    neighbouring frames through the temporal convolutions.

    Args:
        n_joints:    Number of keypoints in this stream (21 or 12).
        hidden_dim:  Channel width throughout.
        conv_layers: Number of temporal conv blocks.
        kernel_size: Conv kernel size (odd).
        dropout:     Dropout rate.
    """

    def __init__(
        self,
        n_joints: int,
        n_dims: int = 3,
        hidden_dim: int = 256,
        conv_layers: int = 3,
        kernel_size: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.frame_proj = nn.Sequential(
            nn.Linear(n_joints * n_dims, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.temporal_convs = nn.Sequential(
            *[TemporalConvBlock(hidden_dim, kernel_size, dropout) for _ in range(conv_layers)]
        )
        self.out_dim = hidden_dim

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, J, 3)  ->  (B, T, D)"""
        B, T, J, C = x.shape
        valid  = ~torch.isnan(x[:, :, 0, 0])          # (B, T) — wrist must be present
        x_flat = torch.nan_to_num(x.reshape(B, T, J * C), nan=0.0)
        feat   = self.frame_proj(x_flat)               # (B, T, D)
        feat   = self.temporal_convs(feat.transpose(1, 2)).transpose(1, 2)  # (B, T, D)
        feat   = feat * valid.unsqueeze(-1)             # zero invalid frames
        return feat
