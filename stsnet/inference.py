"""
Clean inference API for STS-Net.

Provides STSNetInference for per-frame prediction and Viterbi alignment.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from stsnet.model import STSNet
from stsnet.data.pose_io import load_pose_streams
from stsnet.data.description import _parse_description
from stsnet.viterbi import (
    ctc_forced_align,
    frame_labels_to_inner_segs,
    equal_spacing_inner,
    HEURISTIC_PARAMS,
    PREP_LABEL,
    RETRACT_LABEL,
)


class STSNetInference:
    """
    High-level inference API for a trained STS-Net checkpoint.

    Args:
        checkpoint_path: Path to a best.pt checkpoint saved by scripts/train.py.
        device:          Torch device string (e.g. "cpu", "cuda:0").
    """

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu"):
        self.device = torch.device(device)
        self.model, self.vocabs = STSNet.from_checkpoint(
            str(checkpoint_path), map_location=device
        )
        self.model.to(self.device)
        self.model.eval()

        # Build reverse mappings (idx → label)
        self._idx_to_shape = {v: k for k, v in self.vocabs["shape_to_idx"].items()}
        self._idx_to_att   = {v: k for k, v in self.vocabs["att_to_idx"].items()}

        motion_to_idx = self.vocabs.get("motion_to_idx", {})
        self._idx_to_motion = {v: k for k, v in motion_to_idx.items()}

        state_to_idx = self.vocabs.get("state_to_idx", {})
        self._idx_to_state = {v: k for k, v in state_to_idx.items()}

        cloc_to_idx = self.vocabs.get("cloc_to_idx", {})
        self._idx_to_cloc  = {v: k for k, v in cloc_to_idx.items()}

        ctype_to_idx = self.vocabs.get("ctype_to_idx", {})
        self._idx_to_ctype = {v: k for k, v in ctype_to_idx.items()}

    def _load_streams(
        self, pose_path: str | Path, handedness: str = "right"
    ) -> dict[str, torch.Tensor] | None:
        """Load and normalize pose streams, returning tensors on self.device."""
        streams = load_pose_streams(Path(pose_path), handedness, mirror_left=True)
        if streams is None:
            return None
        return {
            k: torch.from_numpy(v).unsqueeze(0).to(self.device)   # (1, T, J, 3)
            for k, v in streams.items()
        }

    def _run_model(
        self, streams: dict[str, torch.Tensor]
    ) -> dict[str, np.ndarray]:
        """Run model forward pass, returning log-softmax arrays per head."""
        with torch.no_grad():
            out = self.model(
                streams["dominant"],
                streams["nondominant"],
                streams["body"],
                face=streams["face"],
            )
        head_map = {
            "state":        "state_logits",
            "shape":        "shape_logits",
            "att":          "att_logits",
            "hand_type":    "hand_type_logits",
            "motion":       "motion_logits",
            "contact_loc":  "contact_loc_logits",
            "contact_type": "contact_type_logits",
            "nondom_shape": "nondom_shape_logits",
            "nondom_att":   "nondom_att_logits",
        }
        return {
            h: F.log_softmax(out[lk], dim=-1).squeeze(0).cpu().numpy()
            for h, lk in head_map.items() if lk in out
        }

    def predict_clip(
        self, pose_path: str | Path, handedness: str = "right"
    ) -> dict[str, np.ndarray]:
        """
        Run the model on a pose file and return per-frame integer class indices
        for all heads.

        Returns:
            dict mapping head name → (T,) int32 array of argmax indices.
            Returns empty dict if pose loading fails.
        """
        streams = self._load_streams(pose_path, handedness)
        if streams is None:
            return {}
        log_probs = self._run_model(streams)
        return {h: lp.argmax(axis=-1).astype(np.int32) for h, lp in log_probs.items()}

    def predict_clip_decoded(
        self, pose_path: str | Path, handedness: str = "right"
    ) -> dict[str, list[str]]:
        """
        Run the model and return per-frame label strings for all heads.

        Returns:
            dict mapping head name → list of T label strings.
        """
        preds = self.predict_clip(pose_path, handedness)
        if not preds:
            return {}

        rev_maps = {
            "state":        self._idx_to_state,
            "shape":        self._idx_to_shape,
            "att":          self._idx_to_att,
            "motion":       self._idx_to_motion,
            "contact_loc":  self._idx_to_cloc,
            "contact_type": self._idx_to_ctype,
            "nondom_shape": self._idx_to_shape,
            "nondom_att":   self._idx_to_att,
            "hand_type":    {0: "one", 1: "two"},
        }
        result = {}
        for head, indices in preds.items():
            rev = rev_maps.get(head, {})
            result[head] = [rev.get(int(i), str(i)) for i in indices]
        return result

    def align_clip(
        self,
        pose_path:   str | Path,
        description: str,
        sign_start:  int,
        sign_end:    int,
        handedness:  str = "right",
        min_dur:     int = 3,
        weights:     Optional[dict[str, float]] = None,
    ) -> list[tuple[str, int, int]]:
        """
        Viterbi alignment of a clip against its description.

        Parses the description, runs the model, applies heuristic outer bounds
        (prep/retract), then runs blank-free Viterbi on the inner shapes.

        Args:
            pose_path:   Path to .pose file.
            description: SSLL description string (e.g. "Flata handen, framåtriktad...").
            sign_start:  First frame of signing window (from pseudo_signing.json).
            sign_end:    Last frame (exclusive) of signing window.
            handedness:  "right" or "left".
            min_dur:     Minimum frames per shape segment in Viterbi.
            weights:     Emission head weights dict. Defaults to standard recipe weights.

        Returns:
            List of (label, start_frame, end_frame) tuples covering the full clip.
            Returns an empty list if pose loading or description parsing fails.
        """
        if weights is None:
            weights = {
                "state": 1.0, "shape": 1.0, "att": 0.7,
                "hand_type": 0.3, "motion": 0.5,
                "contact_loc": 0.0, "contact_type": 0.0,
            }

        # Parse description
        phases_raw = _parse_description(description)
        if not phases_raw:
            return []

        shape_to_idx = self.vocabs["shape_to_idx"]
        att_to_idx   = self.vocabs["att_to_idx"]
        shape_fuzzy  = {k.lower().replace(" ", "").replace("-", ""): k
                        for k in shape_to_idx}

        def match(raw):
            if raw in shape_to_idx: return raw
            return shape_fuzzy.get(raw.lower().replace(" ", "").replace("-", ""))

        phases = []
        shapes_list = []
        for tup in phases_raw:
            shape, att, hand_type, cloc, ctype, motion, nd_shape, nd_att = tup
            canonical = match(shape)
            if canonical is None:
                continue
            nd_canonical = match(nd_shape) if nd_shape else None
            phases.append((canonical, att, hand_type, cloc, ctype, motion, nd_canonical, nd_att))
            shapes_list.append(canonical)

        if not phases:
            return []

        # Load pose and run model (crop to signing window)
        streams = load_pose_streams(Path(pose_path), handedness, mirror_left=True)
        if streams is None:
            return []

        full_len = streams["dominant"].shape[0]
        s0, s1   = sign_start, sign_end

        def crop(arr):
            t = torch.from_numpy(arr[s0:s1]).unsqueeze(0).to(self.device)
            return t

        stream_tensors = {
            "dominant":    crop(streams["dominant"]),
            "nondominant": crop(streams["nondominant"]),
            "body":        crop(streams["body"]),
            "face":        crop(streams["face"]),
        }
        log_probs = self._run_model(stream_tensors)  # (T_win, C_head)

        # Heuristic outer bounds
        W   = sign_end - sign_start
        mid = sign_start + W // 2
        prep_end      = sign_start + max(2, round(W * HEURISTIC_PARAMS["prep_frac"]))
        prep_end      = min(prep_end, mid)
        retract_start = sign_end - max(2, HEURISTIC_PARAMS["retract_dur"])
        retract_start = max(retract_start, mid)

        # Inner Viterbi
        inner_s   = prep_end      - sign_start
        inner_e   = retract_start - sign_start
        inner_len = inner_e - inner_s
        L         = len(phases)

        from stsnet.data.align_dataset import build_emission
        from stsnet.data.contact import CONTACT_LOCATIONS, CONTACT_TYPES
        from stsnet.data.description import MOTION_DIRECTIONS

        motion_to_idx = {d: i for i, d in enumerate(MOTION_DIRECTIONS)}
        cloc_to_idx   = {l: i for i, l in enumerate(CONTACT_LOCATIONS)}
        ctype_to_idx  = {t: i for i, t in enumerate(CONTACT_TYPES)}

        if L == 0 or inner_len < L * min_dur:
            inner_segs = equal_spacing_inner(shapes_list, prep_end, retract_start)
        else:
            lp_inner = {h: lp[inner_s:inner_e] for h, lp in log_probs.items()}
            em = build_emission(
                lp_inner, phases,
                shape_to_idx, att_to_idx, motion_to_idx,
                cloc_to_idx, ctype_to_idx, weights,
            )
            frame_labels = ctc_forced_align(em, list(range(L)), blank=0, min_dur=min_dur)
            inner_segs   = frame_labels_to_inner_segs(frame_labels, shapes_list, prep_end)

        # Assemble full segment list
        segs: list[tuple[str, int, int]] = []
        if sign_start > 0:
            segs.append(("rest", 0, sign_start))
        if prep_end > sign_start:
            segs.append((PREP_LABEL, sign_start, prep_end))
        segs.extend(inner_segs)
        if retract_start < sign_end:
            segs.append((RETRACT_LABEL, retract_start, sign_end))
        if sign_end < full_len:
            segs.append(("rest", sign_end, full_len))

        return segs
