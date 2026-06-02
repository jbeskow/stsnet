"""
Inference APIs for STS-Net.

STSNetInference (v0.1): per-frame prediction and Viterbi alignment.
ClipClassifierInference (v0.2): clip-level phonological prediction and embedding.
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


# ---------------------------------------------------------------------------
# ClipClassifierInference  (STS-Net v0.2)
# ---------------------------------------------------------------------------

class ClipClassifierInference:
    """
    High-level inference API for a trained ClipClassifier checkpoint.

    Args:
        checkpoint_path: Path to a best.pt checkpoint saved by scripts/train_clip.py.
        device:          Torch device string (e.g. "cpu", "cuda:0").
    """

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu"):
        from stsnet.clip_classifier import ClipClassifier
        self.device = torch.device(device)
        self.model, self.vocab_meta = ClipClassifier.from_checkpoint(
            str(checkpoint_path), map_location=device
        )
        self.model.to(self.device)
        self.model.eval()

        self._idx_to_shape  = {v: k for k, v in self.vocab_meta["shape_to_idx"].items()}
        self._idx_to_att    = {v: k for k, v in self.vocab_meta["att_to_idx"].items()}
        self._idx_to_motion = {v: k for k, v in self.vocab_meta.get("motion_to_idx", {}).items()}
        self._idx_to_cloc   = {v: k for k, v in self.vocab_meta.get("cloc_to_idx", {}).items()}
        self._idx_to_ctype  = {v: k for k, v in self.vocab_meta.get("ctype_to_idx", {}).items()}

    def _load_streams(
        self,
        pose_path:   str | Path,
        handedness:  str = "right",
        f0:          int = 0,
        f1:          int | None = None,
    ) -> dict[str, torch.Tensor] | None:
        """Load pose streams, optionally sliced to [f0, f1), as (1, T, J, 3) tensors."""
        streams = load_pose_streams(Path(pose_path), handedness, mirror_left=True)
        if streams is None:
            return None
        if f1 is not None:
            streams = {k: v[f0:f1] for k, v in streams.items()}
        elif f0 > 0:
            streams = {k: v[f0:] for k, v in streams.items()}
        return {
            k: torch.from_numpy(v).unsqueeze(0).to(self.device)
            for k, v in streams.items()
        }

    def predict_phonology(
        self,
        pose_path:   str | Path,
        sign_start:  int | None = None,
        sign_end:    int | None = None,
        handedness:  str = "right",
    ) -> dict[str, str] | None:
        """
        Predict phonological properties for a single sign.

        If sign_start/sign_end are provided the model's attention is masked to that
        window; otherwise attention spans the whole clip.

        Returns:
            dict with keys: shape, att, cloc, ctype, motion, hand_type,
                            nondom_shape (if available)
            Returns None if pose loading fails.
        """
        streams = self._load_streams(pose_path, handedness)
        if streams is None:
            return None

        T = streams["dominant"].shape[1]
        kwargs = {}
        if sign_start is not None and sign_end is not None:
            kwargs["sign_start"] = torch.tensor([sign_start], device=self.device)
            kwargs["sign_end"]   = torch.tensor([sign_end],   device=self.device)
            kwargs["lengths"]    = torch.tensor([T],          device=self.device)

        with torch.no_grad():
            out = self.model(
                streams["dominant"], streams["nondominant"],
                streams["body"], streams.get("face"),
                **kwargs,
            )

        rev_maps = {
            "shape_logits":     self._idx_to_shape,
            "att_logits":       self._idx_to_att,
            "motion_logits":    self._idx_to_motion,
            "cloc_logits":      self._idx_to_cloc,
            "ctype_logits":     self._idx_to_ctype,
            "hand_type_logits": {0: "one", 1: "two"},
        }
        result = {}
        key_names = {
            "shape_logits":      "shape",
            "att_logits":        "att",
            "cloc_logits":       "cloc",
            "ctype_logits":      "ctype",
            "motion_logits":     "motion",
            "hand_type_logits":  "hand_type",
            "nondom_shape_logits": "nondom_shape",
        }
        for logit_key, name in key_names.items():
            if logit_key not in out:
                continue
            idx = int(out[logit_key].squeeze(0).argmax())
            rev = rev_maps.get(logit_key, {})
            result[name] = rev.get(idx, str(idx))
        return result

    def embed_clip(
        self,
        pose_path:   str | Path,
        sign_start:  int | None = None,
        sign_end:    int | None = None,
        handedness:  str = "right",
    ) -> np.ndarray | None:
        """
        Return the 256-dim clip embedding for a sign, useful for retrieval or clustering.

        If sign_start/sign_end are given, slice the pose to that window (the
        attention mask is then set to cover all frames of the slice).
        Returns None if pose loading fails.
        """
        if sign_start is not None and sign_end is not None:
            streams = self._load_streams(pose_path, handedness, sign_start, sign_end)
        else:
            streams = self._load_streams(pose_path, handedness)
        if streams is None:
            return None

        T = streams["dominant"].shape[1]
        with torch.no_grad():
            out = self.model(
                streams["dominant"], streams["nondominant"],
                streams["body"], streams.get("face"),
                sign_start=torch.zeros(1, dtype=torch.long, device=self.device),
                sign_end=torch.tensor([T], dtype=torch.long, device=self.device),
                lengths=torch.tensor([T], dtype=torch.long, device=self.device),
            )
        return out["clip_emb"].squeeze(0).cpu().numpy()
