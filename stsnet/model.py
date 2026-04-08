"""
STS-Net: per-frame multi-head phonology model.

Nine output heads trained with cross-entropy (ignore_index=-1):
  state       — rest / prep / sign / retract   (4 classes)
  shape       — dominant handshape             (vocab size)
  att         — attitude                       (vocab size)
  hand_type   — one / two                      (2 classes)
  contact_loc — contact location               (vocab size)
  contact_type— contact type                   (vocab size)
  motion      — motion direction               (7 classes)
  nondom_shape— non-dominant handshape         (vocab size, optional)
  nondom_att  — non-dominant attitude          (vocab size, optional)

Architecture:
  - Per-stream FrameEncoder (CNN, residual blocks)
  - Fusion linear over selected streams
  - Optional BiLSTM
  - One linear head per output
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from stsnet.encoder import FrameEncoder


class STSNet(nn.Module):
    """
    Args:
        num_shapes:        dominant handshape vocab size
        num_atts:          attitude vocab size
        num_contact_locs:  contact location vocab size
        num_contact_types: contact type vocab size
        num_nondom_shapes: non-dominant shape vocab size (0 = no head)
        num_nondom_atts:   non-dominant att vocab size   (0 = no head)
        hidden_dim:        encoder channel width
        conv_layers:       temporal conv blocks per encoder
        kernel_size:       conv kernel size
        dropout:           dropout rate
        bilstm_layers:     0 = no BiLSTM; >0 = number of BiLSTM layers
        streams:           which input streams to use; subset of
                           {"dom", "nondom", "body", "face"}
                           (dom is always included)
    """

    STREAM_JOINTS = {"dom": 21, "nondom": 21, "body": 12, "face": 25}

    def __init__(
        self,
        num_shapes:        int,
        num_atts:          int,
        num_contact_locs:  int,
        num_contact_types: int,
        num_nondom_shapes: int   = 0,
        num_nondom_atts:   int   = 0,
        hidden_dim:   int   = 256,
        conv_layers:  int   = 3,
        kernel_size:  int   = 5,
        dropout:      float = 0.2,
        bilstm_layers:int   = 1,
        streams: tuple[str, ...] = ("dom", "nondom", "body", "face"),
    ):
        super().__init__()
        streams = tuple(dict.fromkeys(["dom"] + [s for s in streams if s != "dom"]))
        self.streams = streams

        enc_kw = dict(hidden_dim=hidden_dim, conv_layers=conv_layers,
                      kernel_size=kernel_size, dropout=dropout)
        self.encoders = nn.ModuleDict({
            s: FrameEncoder(n_joints=self.STREAM_JOINTS[s], **enc_kw)
            for s in streams
        })

        D = hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(len(streams) * D, D),
            nn.LayerNorm(D),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.bilstm_layers = bilstm_layers
        if bilstm_layers > 0:
            self.bilstm = nn.LSTM(D, D // 2, num_layers=bilstm_layers,
                                  bidirectional=True, batch_first=True,
                                  dropout=dropout if bilstm_layers > 1 else 0.0)
            self.bilstm_drop = nn.Dropout(dropout)

        # Heads
        self.state_head        = nn.Linear(D, 4)                   # rest/prep/sign/retract
        self.shape_head        = nn.Linear(D, num_shapes)
        self.att_head          = nn.Linear(D, num_atts)
        self.hand_type_head    = nn.Linear(D, 2)                   # one/two
        self.contact_loc_head  = nn.Linear(D, num_contact_locs)
        self.contact_type_head = nn.Linear(D, num_contact_types)
        self.motion_head       = nn.Linear(D, 7)                   # 7 canonical directions

        self.num_nondom_shapes = num_nondom_shapes
        if num_nondom_shapes > 0:
            self.nondom_shape_head = nn.Linear(D, num_nondom_shapes)
        self.num_nondom_atts = num_nondom_atts
        if num_nondom_atts > 0:
            self.nondom_att_head = nn.Linear(D, num_nondom_atts)

    def forward(
        self,
        dominant:    Tensor,              # (B, T, 21, 3)
        nondominant: Tensor,              # (B, T, 21, 3)
        body:        Tensor,              # (B, T, 12, 3)
        face:        Tensor | None = None,# (B, T, 25, 3)
        lengths:     Tensor | None = None,# (B,) for BiLSTM packing
    ) -> dict[str, Tensor]:

        stream_inputs = {
            "dom":    dominant,
            "nondom": nondominant,
            "body":   body,
            "face":   face if face is not None else torch.full(
                (*dominant.shape[:2], 25, 3), float("nan"),
                device=dominant.device, dtype=dominant.dtype),
        }
        feats = [self.encoders[s](stream_inputs[s]) for s in self.streams]
        ctx = self.fusion(torch.cat(feats, dim=-1))   # (B, T, D)

        if self.bilstm_layers > 0:
            if lengths is not None:
                packed = pack_padded_sequence(ctx, lengths.cpu(), batch_first=True,
                                             enforce_sorted=False)
                ctx, _ = self.bilstm(packed)
                ctx, _ = pad_packed_sequence(ctx, batch_first=True,
                                            total_length=dominant.shape[1])
            else:
                ctx, _ = self.bilstm(ctx)
            ctx = self.bilstm_drop(ctx)

        out = {
            "state_logits":        self.state_head(ctx),
            "shape_logits":        self.shape_head(ctx),
            "att_logits":          self.att_head(ctx),
            "hand_type_logits":    self.hand_type_head(ctx),
            "contact_loc_logits":  self.contact_loc_head(ctx),
            "contact_type_logits": self.contact_type_head(ctx),
            "motion_logits":       self.motion_head(ctx),
        }
        if self.num_nondom_shapes > 0:
            out["nondom_shape_logits"] = self.nondom_shape_head(ctx)
        if self.num_nondom_atts > 0:
            out["nondom_att_logits"] = self.nondom_att_head(ctx)
        return out

    @classmethod
    def from_checkpoint(cls, path, map_location="cpu"):
        """Load a standard STS-Net checkpoint, returning (model, vocabs)."""
        ckpt = torch.load(path, map_location=map_location)
        a = ckpt["args"]
        model = cls(
            num_shapes=len(ckpt["vocabs"]["shape_to_idx"]),
            num_atts=len(ckpt["vocabs"]["att_to_idx"]),
            num_contact_locs=len(ckpt["vocabs"].get("cloc_to_idx", {})),
            num_contact_types=len(ckpt["vocabs"].get("ctype_to_idx", {})),
            num_nondom_shapes=len(ckpt["vocabs"]["shape_to_idx"]),
            num_nondom_atts=len(ckpt["vocabs"]["att_to_idx"]),
            hidden_dim=a["hidden_dim"], conv_layers=a["conv_layers"],
            kernel_size=a["kernel_size"], dropout=a.get("dropout", 0.2),
            bilstm_layers=a["bilstm_layers"], streams=tuple(a["streams"]),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        return model, ckpt["vocabs"]
