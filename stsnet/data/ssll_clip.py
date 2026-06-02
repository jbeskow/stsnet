"""
SSLL clip-level multi-label dataset for ClipClassifier (STS-Net v0.2).

Labels are derived from _parse_description(), giving one tuple per signing phase:
    (shape, att, hand_type, cloc, ctype, motion, nondom_shape, nondom_att)

For clips with N phases (handform changes), all features that appear across
any phase are set in the multi-hot target vector (BCE loss).

Sign-frame range comes from pseudo_signing.json (precomputed for each clip).
The model attention pool is masked to these sign frames only.

No alignment CSV is required.
"""

import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from stsnet.data.pose_io import (
    load_pose_streams,
    load_wilor_streams,
    normalize_hand_moryossef,
)
from stsnet.data.contact import CONTACT_LOCATIONS, CONTACT_TYPES
from stsnet.data.description import _parse_description

# Clip-level motion vocab — 'none' at index 0 used as the no-motion fallback
MOTION_DIRECTIONS: list[str] = [
    "none", "nedåt", "uppåt", "framåt", "bakåt", "åt_höger", "åt_vänster", "inåt"
]


def load_sapiens_streams(
    npz_path: Path,
    handedness: str,
    mirror_left: bool = True,
    score_thresh: float = 0.1,
) -> dict[str, np.ndarray] | None:
    """
    Load Sapiens COCO-WholeBody .npz → four streams matching load_pose_streams() output.

    COCO-WholeBody layout (133 kps):
      0-16:   body  (0=nose, 5-6=shoulders, 7-10=elbows/wrists, …)
      23-90:  face  (68 landmarks)
      91-111: left hand  (0=wrist, 1-4=thumb, 5-8=index, …)
      112-132: right hand (same)

    Returns: dominant/nondominant (T,21,3), body (T,17,3), face (T,68,3)
    """
    try:
        npz = np.load(npz_path)
    except Exception:
        return None

    kp = npz["keypoints"].astype(np.float32)   # (T, 133, 2) normalised [0,1]
    sc = npz["scores"].astype(np.float32)       # (T, 133)
    W, H = int(npz["wh"][0]), int(npz["wh"][1])

    xy = kp.copy()
    xy[:, :, 0] *= W
    xy[:, :, 1] *= H
    z   = np.zeros((*xy.shape[:2], 1), dtype=np.float32)
    xyz = np.concatenate([xy, z], axis=-1)   # (T, 133, 3)
    xyz[sc < score_thresh] = np.nan

    body       = xyz[:, 0:17,    :].copy()
    face       = xyz[:, 23:91,   :].copy()
    left_hand  = xyz[:, 91:112,  :].copy()
    right_hand = xyz[:, 112:133, :].copy()

    left_hand  = left_hand  - left_hand[:,  0:1, :]
    right_hand = right_hand - right_hand[:, 0:1, :]

    l_sh = body[:, 5, :]
    r_sh = body[:, 6, :]
    midpoint = np.nanmedian((l_sh + r_sh) / 2.0, axis=0)
    widths   = np.sqrt(np.nansum((r_sh - l_sh)[:, :2] ** 2, axis=1))
    sw = float(np.nanmedian(widths))

    if not (np.isnan(sw) or sw < 1e-6):
        body       = (body      - midpoint) / sw
        face       = (face      - midpoint) / sw
        left_hand  =  left_hand             / sw
        right_hand =  right_hand            / sw

    dominant, nondominant = (
        (right_hand, left_hand) if handedness == "right" else (left_hand, right_hand)
    )

    if handedness == "left" and mirror_left:
        for arr in (dominant, nondominant, body, face):
            arr[..., 0] *= -1.0

    return {"dominant": dominant, "nondominant": nondominant, "body": body, "face": face}


# ── Temporal augmentation ─────────────────────────────────────────────────────

def _time_stretch(v: np.ndarray, T_new: int) -> np.ndarray:
    T = v.shape[0]
    if T_new == T:
        return v
    xi = np.linspace(0, T - 1, T_new)
    lo = np.floor(xi).astype(int).clip(0, T - 2)
    hi = lo + 1
    w  = (xi - lo).reshape((-1,) + (1,) * (v.ndim - 1))
    return ((1.0 - w) * v[lo] + w * v[hi]).astype(v.dtype)


def time_stretch_streams(streams: dict, rate: float) -> dict:
    """Resample all numpy stream arrays by rate along axis 0."""
    T = next(iter(streams.values())).shape[0]
    T_new = max(1, round(T * rate))
    if T_new == T:
        return streams
    return {k: _time_stretch(v, T_new) for k, v in streams.items()}


def jitter_sign_window(sign_start: int, sign_end: int, T: int,
                       max_shift: int) -> tuple[int, int]:
    """Randomly shift the sign window by up to ±max_shift frames."""
    if max_shift == 0:
        return sign_start, sign_end
    shift = random.randint(-max_shift, max_shift)
    new_start = max(0, min(sign_start + shift, T - 1))
    new_end   = max(new_start + 1, min(sign_end + shift, T))
    return new_start, new_end


class SSLLClipDataset(Dataset):
    """
    Clip-level multi-label dataset for isolated SSLL signs.

    Args:
        csv_path:           SSLL metadata CSV
        pose_dir:           directory of .pose files
        vocab_file:         sts_handformer.txt (dominant handshape vocab)
        pseudo_json:        pseudo_signing.json (sign start/end per clip)
        mirror_left:        mirror left-handed signers to right
        noise_std:          Gaussian pose augmentation (training only)
        signers:            restrict to these signer IDs (None = all)
        wilor_dir:          directory of WiLoR .npz files (None = disabled)
        dino_dir:           directory of DINOv2 .npz files (None = disabled)
        sapiens_dir:        directory of Sapiens .npz files; replaces MediaPipe when set
        n_coords:           coordinate dims (3 = xyz, 2 = xy)
        time_stretch_range: (min, max) stretch factors for training augmentation
        temporal_jitter:    max frame shift for sign boundary jitter (0 = disabled)
    """

    HANDEDNESS = "right"

    def __init__(
        self,
        csv_path:           str | Path,
        pose_dir:           str | Path,
        vocab_file:         str | Path,
        pseudo_json:        str | Path,
        mirror_left:        bool = True,
        noise_std:          float = 0.0,
        signers:            list[str] | None = None,
        wilor_dir:          str | Path | None = None,
        dino_dir:           str | Path | None = None,
        sapiens_dir:        str | Path | None = None,
        n_coords:           int = 3,
        time_stretch_range: tuple[float, float] = (1.0, 1.0),
        temporal_jitter:    int = 0,
    ):
        self.pose_dir           = Path(pose_dir)
        self.mirror_left        = mirror_left
        self.noise_std          = noise_std
        self.n_coords           = n_coords
        self.time_stretch_range = time_stretch_range
        self.temporal_jitter    = temporal_jitter
        self.wilor_dir   = Path(wilor_dir)   if wilor_dir   else None
        self.dino_dir    = Path(dino_dir)    if dino_dir    else None
        self.sapiens_dir = Path(sapiens_dir) if sapiens_dir else None

        # ── Vocab ─────────────────────────────────────────────────────────────
        shapes = _load_handshape_vocab(vocab_file)
        self.shape_to_idx: dict[str, int] = {s: i for i, s in enumerate(shapes)}
        self._shape_fuzzy = {
            v.lower().replace(" ", "").replace("-", ""): v for v in shapes
        }
        self.num_shapes = len(shapes)

        self.cloc_to_idx  = {l: i for i, l in enumerate(CONTACT_LOCATIONS)}
        self.ctype_to_idx = {t: i for i, t in enumerate(CONTACT_TYPES)}
        self.num_cloc = len(CONTACT_LOCATIONS)
        self.num_ctype = len(CONTACT_TYPES)

        self.motion_to_idx = {d: i for i, d in enumerate(MOTION_DIRECTIONS)}
        self.num_motion = len(MOTION_DIRECTIONS)

        # ── Load pseudo sign ranges ───────────────────────────────────────────
        with open(pseudo_json, encoding="utf-8") as f:
            pseudo: dict[str, dict] = json.load(f)

        # ── Pass 1: parse descriptions, collect att vocab ─────────────────────
        signer_set = set(signers) if signers else None
        att_vocab: dict[str, int] = {}
        parsed: list[dict] = []
        skipped_no_pose = skipped_no_pseudo = skipped_no_desc = 0

        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                movie = (row.get("movie") or "").strip()
                if not movie or movie == "nan":
                    skipped_no_pose += 1
                    continue
                signer = row.get("signer", "")
                if signer_set and signer not in signer_set:
                    continue
                pose_fname = os.path.basename(movie) + ".pose"
                if not (self.pose_dir / pose_fname).exists():
                    skipped_no_pose += 1
                    continue
                if pose_fname not in pseudo:
                    skipped_no_pseudo += 1
                    continue

                phases_raw = _parse_description(row.get("description", ""))
                if not phases_raw:
                    skipped_no_desc += 1
                    continue

                phases = []
                for tup in phases_raw:
                    shape, att, hand_type, cloc, ctype, motion, nondom_shape, nondom_att = tup
                    canonical = self._match_shape(shape)
                    if canonical is None:
                        continue
                    nondom_can = self._match_shape(nondom_shape) if nondom_shape else None
                    phases.append((canonical, att, hand_type, cloc, ctype,
                                   motion, nondom_can, nondom_att))
                    if att is not None:
                        att_vocab.setdefault(att, len(att_vocab))
                    if nondom_att is not None:
                        att_vocab.setdefault(nondom_att, len(att_vocab))

                if not phases:
                    skipped_no_desc += 1
                    continue

                parsed.append({
                    "pose_path":  self.pose_dir / pose_fname,
                    "phases":     phases,
                    "signer":     signer,
                    "word":       row.get("word", ""),
                    "sign_start": pseudo[pose_fname]["sign_start"],
                    "sign_end":   pseudo[pose_fname]["sign_end"],
                })

        # Freeze attitude vocab (sorted for reproducibility)
        att_labels = sorted(att_vocab)
        self.att_to_idx: dict[str, int] = {a: i for i, a in enumerate(att_labels)}
        self.idx_to_att: dict[int, str]  = {i: a for a, i in self.att_to_idx.items()}
        self.num_atts = len(att_labels)

        # ── Pass 2: build multi-hot target tensors ────────────────────────────
        self.samples: list[dict] = []
        for pr in parsed:
            targets = self._make_targets(pr["phases"])
            self.samples.append({
                "pose_path":  pr["pose_path"],
                "sign_start": pr["sign_start"],
                "sign_end":   pr["sign_end"],
                "signer":     pr["signer"],
                "word":       pr["word"],
                **targets,
            })

        print(
            f"SSLLClipDataset: {len(self.samples)} samples"
            f"  |  no_pose={skipped_no_pose}"
            f"  |  no_pseudo={skipped_no_pseudo}"
            f"  |  no_desc={skipped_no_desc}"
            f"\n  vocabs — shapes={self.num_shapes}"
            f"  atts={self.num_atts}"
            f"  motion={self.num_motion}"
            f"  cloc={self.num_cloc}  ctype={self.num_ctype}"
        )

    def _match_shape(self, raw: str | None) -> str | None:
        if raw is None:
            return None
        if raw in self.shape_to_idx:
            return raw
        return self._shape_fuzzy.get(raw.lower().replace(" ", "").replace("-", ""))

    def _make_targets(self, phases: list[tuple]) -> dict[str, torch.Tensor]:
        shape_t        = torch.zeros(self.num_shapes)
        nondom_shape_t = torch.zeros(self.num_shapes)
        att_t          = torch.zeros(self.num_atts)
        nondom_att_t   = torch.zeros(self.num_atts)
        cloc_t         = torch.zeros(self.num_cloc)
        ctype_t        = torch.zeros(self.num_ctype)
        motion_t       = torch.zeros(self.num_motion)
        hand_type_idx  = -1

        for shape, att, hand_type, cloc, ctype, motion, nondom_shape, nondom_att in phases:
            if shape in self.shape_to_idx:
                shape_t[self.shape_to_idx[shape]] = 1.0
            if nondom_shape is not None and nondom_shape in self.shape_to_idx:
                nondom_shape_t[self.shape_to_idx[nondom_shape]] = 1.0
            if att is not None and att in self.att_to_idx:
                att_t[self.att_to_idx[att]] = 1.0
            if nondom_att is not None and nondom_att in self.att_to_idx:
                nondom_att_t[self.att_to_idx[nondom_att]] = 1.0
            if cloc is not None and cloc in self.cloc_to_idx:
                cloc_t[self.cloc_to_idx[cloc]] = 1.0
            if ctype is not None and ctype in self.ctype_to_idx:
                ctype_t[self.ctype_to_idx[ctype]] = 1.0
            if motion is not None and motion in self.motion_to_idx:
                motion_t[self.motion_to_idx[motion]] = 1.0
            if hand_type_idx == -1:
                hand_type_idx = 1 if hand_type == "two" else 0

        if cloc_t.sum() == 0:
            cloc_t[0] = 1.0
        if ctype_t.sum() == 0:
            ctype_t[0] = 1.0
        if motion_t.sum() == 0:
            motion_t[0] = 1.0

        return {
            "shape_target":        shape_t,
            "nondom_shape_target": nondom_shape_t,
            "att_target":          att_t,
            "nondom_att_target":   nondom_att_t,
            "cloc_target":         cloc_t,
            "ctype_target":        ctype_t,
            "motion_target":       motion_t,
            "hand_type":           torch.tensor(hand_type_idx, dtype=torch.long),
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]

        if self.sapiens_dir is not None:
            video_stem = s["pose_path"].stem.removesuffix(".mp4")
            npz_path   = self.sapiens_dir / (video_stem + ".npz")
            streams = load_sapiens_streams(npz_path, self.HANDEDNESS, self.mirror_left)
            if streams is None:
                streams = {k: np.zeros((1, j, 3), dtype=np.float32)
                           for k, j in [("dominant", 21), ("nondominant", 21),
                                        ("body", 17), ("face", 68)]}
        else:
            streams = load_pose_streams(s["pose_path"], self.HANDEDNESS, self.mirror_left)
            if streams is None:
                streams = {k: np.zeros((1, j, 3), dtype=np.float32)
                           for k, j in [("dominant", 21), ("nondominant", 21),
                                        ("body", 12), ("face", 25)]}

        sign_start = s["sign_start"]
        sign_end   = s["sign_end"]

        if self.training:
            lo, hi = self.time_stretch_range
            if lo < hi:
                rate = random.uniform(lo, hi)
                streams = time_stretch_streams(streams, rate)
                sign_start = max(0, round(sign_start * rate))
                sign_end   = max(sign_start + 1, round(sign_end * rate))
            T = next(iter(streams.values())).shape[0]
            sign_start, sign_end = jitter_sign_window(
                sign_start, sign_end, T, self.temporal_jitter)

        item = {k: torch.from_numpy(v) for k, v in streams.items()}

        item["dom_norm"]    = torch.from_numpy(normalize_hand_moryossef(streams["dominant"]))
        item["nondom_norm"] = torch.from_numpy(normalize_hand_moryossef(streams["nondominant"]))

        if self.wilor_dir is not None:
            wilor = load_wilor_streams(s["pose_path"], self.wilor_dir,
                                       self.HANDEDNESS, self.mirror_left)
            T = item["dominant"].shape[0]
            if wilor is None:
                item["wilor_dom"]    = torch.full((T, 21, 3), float("nan"))
                item["wilor_nondom"] = torch.full((T, 21, 3), float("nan"))
            else:
                for wk in ("wilor_dom", "wilor_nondom"):
                    arr = torch.from_numpy(wilor[wk])
                    T_w = arr.shape[0]
                    if T_w > T:
                        arr = arr[:T]
                    elif T_w < T:
                        arr = torch.cat([arr, torch.full((T - T_w, 21, 3), float("nan"))], dim=0)
                    item[wk] = arr

        if self.dino_dir is not None:
            T = item["dominant"].shape[0]
            stem = Path(s["pose_path"]).stem.removesuffix(".pose")
            dino_path = self.dino_dir / (stem + ".npz")
            if dino_path.exists():
                dino_feat = np.load(dino_path)["features"].astype(np.float32)  # (T_d, 1024)
                T_d = dino_feat.shape[0]
                if T_d > T:
                    dino_feat = dino_feat[:T]
                elif T_d < T:
                    dino_feat = np.concatenate(
                        [dino_feat, np.zeros((T - T_d, dino_feat.shape[1]), dtype=np.float32)],
                        axis=0)
                item["dino"] = torch.from_numpy(dino_feat)
            else:
                item["dino"] = torch.zeros(T, 1024)

        item["sign_start"] = torch.tensor(sign_start, dtype=torch.long)
        item["sign_end"]   = torch.tensor(sign_end,   dtype=torch.long)
        item["signer"]     = s["signer"]
        item["word"]       = s["word"]

        for key in ("shape_target", "nondom_shape_target", "att_target",
                    "nondom_att_target", "cloc_target", "ctype_target",
                    "motion_target", "hand_type"):
            item[key] = s[key]

        if self.n_coords < 3:
            for k in ("dominant", "nondominant", "body", "face",
                      "dom_norm", "nondom_norm"):
                if k in item:
                    item[k] = item[k][..., :self.n_coords]

        if self.noise_std > 0.0 and self.training:
            for k in ("dominant", "nondominant", "body", "dom_norm", "nondom_norm"):
                if k in item:
                    item[k] = item[k] + torch.randn_like(item[k]) * self.noise_std

        return item

    @property
    def training(self):
        return getattr(self, "_training", True)

    def train(self, mode=True):
        self._training = mode
        return self

    def eval(self):
        return self.train(False)


def collate_clip(batch: list[dict]) -> dict:
    """Pad pose streams to longest sequence; stack fixed-size targets."""
    str_keys    = {"signer", "word"}
    scalar_keys = {"sign_start", "sign_end", "hand_type"}
    target_keys = {"shape_target", "nondom_shape_target", "att_target",
                   "nondom_att_target", "cloc_target", "ctype_target", "motion_target"}
    fixed_keys  = str_keys | scalar_keys | target_keys

    all_keys = set(batch[0])
    for b in batch[1:]:
        all_keys &= set(b)

    stream_keys = {k for k in all_keys if k not in fixed_keys
                   and isinstance(batch[0][k], torch.Tensor)
                   and batch[0][k].dim() == 3}
    flat_keys   = {k for k in all_keys if k not in fixed_keys
                   and isinstance(batch[0][k], torch.Tensor)
                   and batch[0][k].dim() == 2}

    out = {k: [b[k] for b in batch] for k in str_keys}
    T_max = max(b["dominant"].shape[0] for b in batch)

    for k in stream_keys:
        J, C = batch[0][k].shape[1], batch[0][k].shape[2]
        padded = torch.full((len(batch), T_max, J, C), float("nan"))
        for i, b in enumerate(batch):
            t = b[k].shape[0]
            padded[i, :t] = b[k]
        out[k] = padded

    for k in flat_keys:
        D = batch[0][k].shape[1]
        padded = torch.zeros(len(batch), T_max, D)
        for i, b in enumerate(batch):
            t = b[k].shape[0]
            padded[i, :t] = b[k]
        out[k] = padded

    out["lengths"] = torch.tensor([b["dominant"].shape[0] for b in batch])

    for k in scalar_keys:
        out[k] = torch.stack([b[k] for b in batch])

    for k in target_keys:
        out[k] = torch.stack([b[k] for b in batch])

    return out


# ── Vocab helper ──────────────────────────────────────────────────────────────

def _load_handshape_vocab(vocab_file: str | Path) -> list[str]:
    """Load handshape labels from sts_handformer.txt (one label per line)."""
    with open(vocab_file, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
