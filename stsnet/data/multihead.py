"""
SSLL per-frame multi-head dataset for STS-Net.

Produces per-frame labels for nine heads:
  state       : rest(0) / prep(1) / sign(2) / retract(3)
  shape       : dominant handshape (vocab index)
  att         : attitude slug (vocab index)
  hand_type   : one(0) / two(1)
  contact_loc : contact location (vocab index)
  contact_type: contact type (vocab index)
  motion      : motion direction (vocab index)
  nondom_shape: non-dominant handshape (vocab index)
  nondom_att  : non-dominant attitude (vocab index)

Label assignment
----------------
Requires an alignment CSV (from ctc_align.py / stsnet_align.py) that gives
the full per-frame segment sequence for each clip:

  [rest | __prep__ | shape_A | shape_B | ... | __retract__ | rest]

The description for that clip is parsed by _parse_description into a flat list
of phase tuples, one per signing phase (including sub-phases from "förändras
till").  Signing segments are matched to description tuples positionally; clips
where the counts disagree are skipped.

  state  ← segment label (rest/prep/sign/retract)
  shape, att, hand_type, contact_loc, contact_type, motion,
  nondom_shape, nondom_att  ← from matched description tuple

  prep frames   → labels from description tuple 0 (first signing phase)
  retract frames → labels from description tuple -1 (last signing phase)
  rest frames   → all labels = ignore_index (-1)

All heads use ignore_index=-1.  The model uses nn.CrossEntropyLoss(ignore_index=-1).
"""

import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from stsnet.data.pose_io import load_pose_streams
from stsnet.data.contact import CONTACT_LOCATIONS, CONTACT_TYPES
from stsnet.data.description import (
    load_handshape_vocab,
    _normalise_shape,
    _parse_description,
    MOTION_DIRECTIONS,
)

# ── State vocab ────────────────────────────────────────────────────────────────
STATE_VOCAB = ["rest", "prep", "sign", "retract"]
STATE_TO_IDX = {s: i for i, s in enumerate(STATE_VOCAB)}
IGNORE = -1
_NON_SIGN_LABELS = {"rest", "<blank>", "__prep__", "__retract__"}


# ── Alignment loading ──────────────────────────────────────────────────────────

def load_segments(
    alignment_csv: str | Path,
) -> dict[str, list[tuple[str, int, int]]]:
    """
    Return {pose_file: [(label, start_frame, end_frame), ...]} for dominant hand,
    sorted by start_frame.
    """
    rows: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    with open(alignment_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("hand", "dominant") != "dominant":
                continue
            rows[row["pose_file"]].append((
                row["label"],
                int(row["start_frame"]),
                int(row["end_frame"]),
            ))
    return {k: sorted(v, key=lambda x: x[1]) for k, v in rows.items()}


# ── Frame-label builder ────────────────────────────────────────────────────────

def _make_labels(
    segments:       list[tuple[str, int, int]],
    desc_phases:    list[tuple],          # output of _parse_description, already vocab-filtered
    T:              int,
    shape_to_idx:   dict[str, int],
    att_to_idx:     dict[str, int],
    motion_to_idx:  dict[str, int],
    cloc_to_idx:    dict[str, int],
    ctype_to_idx:   dict[str, int],
) -> dict[str, torch.Tensor] | None:
    """
    Build one tensor per head, shape (T,), dtype=int64, default=IGNORE.

    Returns None if the number of signing segments != number of description phases
    (i.e. the alignment and description are inconsistent for this clip).
    """
    signing_segs = [(l, s, e) for l, s, e in segments
                    if l not in _NON_SIGN_LABELS]

    if len(signing_segs) != len(desc_phases):
        return None

    state_arr  = np.full(T, IGNORE, dtype=np.int64)
    shape_arr  = np.full(T, IGNORE, dtype=np.int64)
    att_arr    = np.full(T, IGNORE, dtype=np.int64)
    ht_arr     = np.full(T, IGNORE, dtype=np.int64)
    cloc_arr   = np.full(T, IGNORE, dtype=np.int64)
    ctype_arr  = np.full(T, IGNORE, dtype=np.int64)
    motion_arr = np.full(T, IGNORE, dtype=np.int64)
    nds_arr    = np.full(T, IGNORE, dtype=np.int64)
    nda_arr    = np.full(T, IGNORE, dtype=np.int64)

    sign_idx = 0  # index into desc_phases for signing segments

    for label, seg_start, seg_end in segments:
        s = max(0, seg_start)
        e = min(T, seg_end)
        if s >= e:
            continue

        if label in ("rest", "<blank>"):
            state_arr[s:e] = STATE_TO_IDX["rest"]
            # All feature heads remain IGNORE for rest frames
            continue

        if label == "__prep__":
            state_arr[s:e] = STATE_TO_IDX["prep"]
            phase = desc_phases[0] if desc_phases else None
        elif label == "__retract__":
            state_arr[s:e] = STATE_TO_IDX["retract"]
            phase = desc_phases[-1] if desc_phases else None
        else:
            state_arr[s:e] = STATE_TO_IDX["sign"]
            phase = desc_phases[sign_idx]
            sign_idx += 1

        if phase is None:
            continue

        shape, att, hand_type, cloc, ctype, motion, nondom_shape, nondom_att = phase

        if shape in shape_to_idx:
            shape_arr[s:e] = shape_to_idx[shape]
        if att is not None and att in att_to_idx:
            att_arr[s:e] = att_to_idx[att]
        ht_arr[s:e] = 1 if hand_type == "two" else 0
        cloc_arr[s:e]  = cloc_to_idx.get(cloc  or "none", 0)
        ctype_arr[s:e] = ctype_to_idx.get(ctype or "none", 0)
        if motion is not None and motion in motion_to_idx:
            motion_arr[s:e] = motion_to_idx[motion]
        if nondom_shape is not None and nondom_shape in shape_to_idx:
            nds_arr[s:e] = shape_to_idx[nondom_shape]
        if nondom_att is not None and nondom_att in att_to_idx:
            nda_arr[s:e] = att_to_idx[nondom_att]

    return {
        "state":       torch.from_numpy(state_arr),
        "shape":       torch.from_numpy(shape_arr),
        "att":         torch.from_numpy(att_arr),
        "hand_type":   torch.from_numpy(ht_arr),
        "contact_loc": torch.from_numpy(cloc_arr),
        "contact_type":torch.from_numpy(ctype_arr),
        "motion":      torch.from_numpy(motion_arr),
        "nondom_shape":torch.from_numpy(nds_arr),
        "nondom_att":  torch.from_numpy(nda_arr),
    }


# ── Dataset ────────────────────────────────────────────────────────────────────

class SSLLMultiHeadDataset(Dataset):
    """
    Per-frame multi-head CE dataset for STS-Net training.

    Args:
        csv_path:      SSLL metadata CSV (sign_data.csv)
        pose_dir:      directory of .pose files
        vocab_file:    sts_handformer.txt  (dominant handshape vocab)
        alignment_csv: alignment CSV from ctc_align / stsnet_align
        mirror_left:   flip left-handed signers to right-handed convention
        noise_std:     Gaussian noise augmentation on pose features
        signers:       optional list of signer IDs to include (None = all)
        llm_cache:     optional path to LLM description-parse cache JSON
        signer_map:    optional path to signer_map.csv (video_id -> signer);
                       if provided, takes precedence over the signer column in
                       csv_path
    """

    HANDEDNESS = "right"

    def __init__(
        self,
        csv_path:      str | Path,
        pose_dir:      str | Path,
        vocab_file:    str | Path,
        alignment_csv: str | Path,
        mirror_left:   bool  = True,
        noise_std:     float = 0.0,
        signers:       list[str] | None = None,
        llm_cache:     str | Path | None = None,
        signer_map:    str | Path | None = None,
    ):
        self.pose_dir    = Path(pose_dir)
        self.mirror_left = mirror_left
        self.noise_std   = noise_std

        if llm_cache is not None:
            from stsnet.data.description import load_llm_parse_cache
            load_llm_parse_cache(llm_cache)

        # ── Vocabs ────────────────────────────────────────────────────────
        shapes = load_handshape_vocab(vocab_file)
        self.shape_to_idx: dict[str, int] = {s: i for i, s in enumerate(shapes)}
        self._shape_fuzzy = {
            v.lower().replace(" ", "").replace("-", ""): v for v in shapes
        }
        self.num_shapes = len(shapes)

        self.state_to_idx  = STATE_TO_IDX
        self.num_states    = len(STATE_VOCAB)

        self.cloc_to_idx   = {l: i for i, l in enumerate(CONTACT_LOCATIONS)}
        self.ctype_to_idx  = {t: i for i, t in enumerate(CONTACT_TYPES)}
        self.num_cloc      = len(CONTACT_LOCATIONS)
        self.num_ctype     = len(CONTACT_TYPES)

        self.motion_to_idx = {d: i for i, d in enumerate(MOTION_DIRECTIONS)}
        self.num_motion    = len(MOTION_DIRECTIONS)

        # Attitude vocab is built from data (two-pass)
        self.att_to_idx:  dict[str, int] = {}
        self.idx_to_att:  dict[int, str] = {}
        self.num_atts = 0

        # ── Load signer map (video_id -> signer) ──────────────────────────
        self._signer_map: dict[str, str] = {}
        if signer_map is not None:
            with open(signer_map, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    self._signer_map[row["video_id"]] = row["signer"]

        # ── Load alignment segments ────────────────────────────────────────
        all_segments = load_segments(alignment_csv)
        print(f"Alignment: {len(all_segments)} clips")

        # ── Pass 1: parse descriptions, collect att vocab ──────────────────
        signer_set = set(signers) if signers else None
        parsed: list[dict] = []
        skipped_no_pose = skipped_no_align = skipped_no_desc = 0
        skipped_mismatch = skipped_no_phases = 0

        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                movie = (row.get("movie") or "").strip()
                if not movie or movie == "nan":
                    skipped_no_pose += 1
                    continue
                # Prefer signer_map; fall back to CSV column
                signer = (self._signer_map.get(movie)
                          or row.get("signer", ""))
                if signer_set and signer not in signer_set:
                    continue
                pose_fname = os.path.basename(movie) + ".pose"
                if not (self.pose_dir / pose_fname).exists():
                    skipped_no_pose += 1
                    continue
                if pose_fname not in all_segments:
                    skipped_no_align += 1
                    continue

                phases_raw = _parse_description(row.get("description", ""))
                if not phases_raw:
                    skipped_no_desc += 1
                    continue

                # Filter phases to those whose shape is in vocab
                phases = []
                for tup in phases_raw:
                    shape, att, hand_type, cloc, ctype, motion, nondom_shape, nondom_att = tup
                    canonical = self._match_shape(shape)
                    if canonical is None:
                        continue
                    nondom_canonical = self._match_shape(nondom_shape) if nondom_shape else None
                    phases.append((canonical, att, hand_type, cloc, ctype,
                                   motion, nondom_canonical, nondom_att))
                    if att is not None:
                        self.att_to_idx.setdefault(att, len(self.att_to_idx))
                    if nondom_att is not None:
                        self.att_to_idx.setdefault(nondom_att, len(self.att_to_idx))

                if not phases:
                    skipped_no_phases += 1
                    continue

                # Quick mismatch check (full check in pass 2 with labels)
                segs = all_segments[pose_fname]
                n_signing = sum(1 for l, _, _ in segs if l not in _NON_SIGN_LABELS)
                if n_signing != len(phases):
                    skipped_mismatch += 1
                    continue

                parsed.append({
                    "pose_path": self.pose_dir / pose_fname,
                    "pose_fname": pose_fname,
                    "phases":    phases,
                    "signer":    signer,
                    "word":      row.get("word", ""),
                })

        # Freeze attitude vocab (sorted for reproducibility)
        att_labels = sorted(self.att_to_idx)
        self.att_to_idx = {a: i for i, a in enumerate(att_labels)}
        self.idx_to_att = {i: a for a, i in self.att_to_idx.items()}
        self.num_atts = len(att_labels)

        # ── Pass 2: build samples with pre-computed label dicts ────────────
        self.samples: list[dict] = []
        for pr in parsed:
            segs   = all_segments[pr["pose_fname"]]
            labels = _make_labels(
                segs, pr["phases"],
                T=segs[-1][2],   # end frame of last segment
                shape_to_idx  = self.shape_to_idx,
                att_to_idx    = self.att_to_idx,
                motion_to_idx = self.motion_to_idx,
                cloc_to_idx   = self.cloc_to_idx,
                ctype_to_idx  = self.ctype_to_idx,
            )
            if labels is None:
                skipped_mismatch += 1
                continue
            self.samples.append({
                "pose_path": pr["pose_path"],
                "labels":    labels,
                "signer":    pr["signer"],
                "word":      pr["word"],
            })

        n = len(self.samples)
        print(
            f"SSLLMultiHeadDataset: {n} samples"
            f"  |  no_pose={skipped_no_pose}"
            f"  |  no_align={skipped_no_align}"
            f"  |  no_desc={skipped_no_desc}"
            f"  |  no_phases={skipped_no_phases}"
            f"  |  seg_mismatch={skipped_mismatch}"
            f"\n  vocabs — shapes={self.num_shapes}"
            f"  atts={self.num_atts}"
            f"  states={self.num_states}"
            f"  motion={self.num_motion}"
            f"  cloc={self.num_cloc}  ctype={self.num_ctype}"
        )

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _match_shape(self, raw: str | None) -> str | None:
        if raw is None:
            return None
        if raw in self.shape_to_idx:
            return raw
        key = raw.lower().replace(" ", "").replace("-", "")
        return self._shape_fuzzy.get(key)

    # ── Dataset protocol ───────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        streams = load_pose_streams(s["pose_path"], self.HANDEDNESS, self.mirror_left)
        if streams is None:
            # Fallback: return zeros if pose failed to load
            T = next(iter(s["labels"].values())).shape[0]
            streams = {k: np.zeros((T, *sh), dtype=np.float32)
                       for k, sh in [("dominant", (21, 3)), ("nondominant", (21, 3)),
                                     ("body", (12, 3)), ("face", (25, 3))]}

        dominant    = torch.from_numpy(streams["dominant"].astype(np.float32))
        nondominant = torch.from_numpy(streams["nondominant"].astype(np.float32))
        body        = torch.from_numpy(streams["body"].astype(np.float32))
        face        = torch.from_numpy(streams["face"].astype(np.float32))

        # Valid mask: frames where dominant wrist is not NaN
        valid = torch.isfinite(dominant[:, 0, 0])

        T_pose = dominant.shape[0]
        T_lbl  = next(iter(s["labels"].values())).shape[0]

        if self.noise_std > 0 and self.training if hasattr(self, "training") else False:
            dominant    = dominant    + torch.randn_like(dominant)    * self.noise_std
            nondominant = nondominant + torch.randn_like(nondominant) * self.noise_std

        # Align label length to pose length (trim or pad with IGNORE)
        def _align(t: torch.Tensor) -> torch.Tensor:
            if T_lbl == T_pose:
                return t
            if T_lbl > T_pose:
                return t[:T_pose]
            pad = torch.full((T_pose - T_lbl,), IGNORE, dtype=torch.long)
            return torch.cat([t, pad])

        labels = {k: _align(v) for k, v in s["labels"].items()}

        return {
            "dominant":    dominant,
            "nondominant": nondominant,
            "body":        body,
            "face":        face,
            "valid":       valid,
            **labels,
            "signer": s["signer"],
            "word":   s["word"],
        }
