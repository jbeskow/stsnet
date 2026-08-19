"""
STSAlignDataset and build_emission for STS-Net Viterbi alignment.

Loads sign windows + description phases + all pose streams for alignment.
"""

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from stsnet.data.pose_io import load_pose_streams
from stsnet.data.contact import CONTACT_LOCATIONS, CONTACT_TYPES
from stsnet.data.description import (
    load_handshape_vocab,
    _parse_description,
    load_llm_parse_cache,
    MOTION_DIRECTIONS,
)
from stsnet.data.multihead import STATE_TO_IDX


# ---------------------------------------------------------------------------
# Sign window loading helpers
# ---------------------------------------------------------------------------

def load_sign_windows_from_pseudo(pseudo_json: str) -> dict[str, tuple[int, int, int]]:
    """
    Load signing windows from a pseudo_signing.json produced by
    scripts/generate_pseudo_signing.py.

    Returns {pose_file: (sign_start, sign_end, full_len)}.
    """
    with open(pseudo_json, encoding="utf-8") as f:
        ps = json.load(f)
    windows: dict[str, tuple[int, int, int]] = {}
    for pose_fname, d in ps.items():
        s, e, T = d["sign_start"], d["sign_end"], d["T"]
        if e > s:
            windows[pose_fname] = (s, e, T)
    return windows


def load_sign_windows(alignment_csv: str) -> dict[str, tuple[int, int, int]]:
    """
    Derive per-clip signing windows from rest segments in the alignment CSV.

    Returns {pose_file: (sign_start, sign_end, full_len)} where
        sign_start = end of first rest segment
        sign_end   = start of last rest segment
        full_len   = max end_frame seen for this clip
    """
    segs:     dict[str, list[dict]] = defaultdict(list)
    max_ends: dict[str, int]        = defaultdict(int)

    with open(alignment_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["hand"] != "dominant":
                continue
            start = int(row["start_frame"])
            end   = int(row["end_frame"])
            pf    = row["pose_file"]
            segs[pf].append({"label": row["label"], "start": start, "end": end})
            max_ends[pf] = max(max_ends[pf], end)

    windows: dict[str, tuple[int, int, int]] = {}
    for pf, rows in segs.items():
        rows  = sorted(rows, key=lambda r: r["start"])
        rests = [r for r in rows if r["label"] == "rest"]
        if len(rests) >= 2:
            s, e = rests[0]["end"], rests[-1]["start"]
            if e > s:
                windows[pf] = (s, e, max_ends[pf])
    return windows


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class STSAlignDataset(Dataset):
    """
    Like SSLLMultiHeadDataset but also returns per-phase description tuples
    needed to build multi-head emission matrices.
    """

    HANDEDNESS = "right"

    def __init__(
        self,
        csv_path:       str | Path,
        pose_dir:       str | Path,
        vocab_file:     str | Path,
        alignment_csv:  str | Path | None = None,
        pseudo_signing: str | Path | None = None,
        llm_cache:      Optional[str] = None,
        mirror_left:    bool = True,
        max_shapes:     int | None = None,
    ):
        self.pose_dir    = Path(pose_dir)
        self.mirror_left = mirror_left
        self.max_shapes  = max_shapes

        if llm_cache:
            load_llm_parse_cache(llm_cache)

        shapes = load_handshape_vocab(vocab_file)
        self._shape_fuzzy = {
            v.lower().replace(" ", "").replace("-", ""): v for v in shapes
        }
        self.shape_to_idx = {s: i for i, s in enumerate(shapes)}

        if pseudo_signing is not None:
            sign_windows = load_sign_windows_from_pseudo(str(pseudo_signing))
            print(f"Sign windows from pseudo_signing: {len(sign_windows)}")
        elif alignment_csv is not None:
            sign_windows = load_sign_windows(str(alignment_csv))
            print(f"Sign windows from alignment_csv: {len(sign_windows)}")
        else:
            raise ValueError("Provide alignment_csv or pseudo_signing")

        self.samples: list[dict] = []
        skipped = defaultdict(int)

        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                movie = (row.get("movie") or "").strip()
                if not movie or movie == "nan":
                    skipped["no_pose"] += 1; continue
                pose_fname = os.path.basename(movie) + ".pose"
                if not (self.pose_dir / pose_fname).exists():
                    skipped["no_pose"] += 1; continue
                if pose_fname not in sign_windows:
                    skipped["no_window"] += 1; continue
                sign_start, sign_end, full_len = sign_windows[pose_fname]

                phases_raw = _parse_description(row.get("description", ""))
                if not phases_raw:
                    skipped["no_label"] += 1; continue

                phases = []
                shapes_list = []
                for tup in phases_raw:
                    shape, att, hand_type, cloc, ctype, motion, nd_shape, nd_att = tup
                    canonical = self._match_shape(shape)
                    if canonical is None:
                        continue
                    nd_canonical = self._match_shape(nd_shape)
                    phases.append((canonical, att, hand_type, cloc, ctype,
                                   motion, nd_canonical, nd_att))
                    shapes_list.append(canonical)

                if not phases:
                    skipped["not_in_vocab"] += 1; continue
                if max_shapes is not None and len(phases) > max_shapes:
                    skipped["too_many_shapes"] += 1; continue
                if sign_end - sign_start < len(phases) * 2 + 1:
                    skipped["window_too_short"] += 1; continue

                self.samples.append({
                    "pose_path":   self.pose_dir / pose_fname,
                    "pose_fname":  pose_fname,
                    "phases":      phases,
                    "shapes_list": shapes_list,
                    "sign_start":  sign_start,
                    "sign_end":    sign_end,
                    "full_len":    full_len,
                    "signer":      row.get("signer", ""),
                    "word":        row.get("word", ""),
                })

        print(f"STSAlignDataset: {len(self.samples)} samples"
              + "".join(f"  |  {v} {k}" for k, v in skipped.items() if v))

    def _match_shape(self, raw: str | None) -> str | None:
        if raw is None: return None
        if raw in self.shape_to_idx: return raw
        return self._shape_fuzzy.get(raw.lower().replace(" ", "").replace("-", ""))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        streams = load_pose_streams(s["pose_path"], self.HANDEDNESS, self.mirror_left)

        nan21 = np.full((1, 21, 3), np.nan, dtype=np.float32)
        nan12 = np.full((1, 12, 3), np.nan, dtype=np.float32)
        nan25 = np.full((1, 25, 3), np.nan, dtype=np.float32)
        if streams is None:
            streams = {"dominant": nan21, "nondominant": nan21,
                       "body": nan12, "face": nan25}

        s0, s1 = s["sign_start"], s["sign_end"]
        dom    = torch.from_numpy(streams["dominant"][s0:s1])
        nondom = torch.from_numpy(streams["nondominant"][s0:s1])
        body   = torch.from_numpy(streams["body"][s0:s1])
        face_np = streams.get("face", nan25)
        face   = torch.from_numpy(
            face_np[s0:s1] if face_np.shape[0] >= s1 else nan25)

        if dom.shape[0] == 0:
            dom = torch.from_numpy(nan21)
            nondom = torch.from_numpy(nan21)
            body = torch.from_numpy(nan12)
            face = torch.from_numpy(nan25)

        return {
            "dominant":    dom,
            "nondominant": nondom,
            "body":        body,
            "face":        face,
            "pose_fname":  s["pose_fname"],
            "sign_start":  s["sign_start"],
            "sign_end":    s["sign_end"],
            "full_len":    s["full_len"],
            "phases":      s["phases"],
            "shapes_list": s["shapes_list"],
            "signer":      s["signer"],
            "word":        s["word"],
        }


def collate_align(batch: list[dict]) -> dict:
    """Pad tensors; keep variable-length fields as lists."""
    tensor_keys = ["dominant", "nondominant", "body", "face"]
    list_keys   = ["pose_fname", "sign_start", "sign_end", "full_len",
                   "phases", "shapes_list", "signer", "word"]
    out = {k: [b[k] for b in batch] for k in list_keys}
    for k in tensor_keys:
        tensors = [b[k] for b in batch]
        T_max = max(t.shape[0] for t in tensors)
        padded = torch.zeros(len(tensors), T_max, *tensors[0].shape[1:])
        for i, t in enumerate(tensors):
            padded[i, :t.shape[0]] = t
        out[k] = padded
    out["lengths"] = torch.tensor([b["dominant"].shape[0] for b in batch])
    return out


# ---------------------------------------------------------------------------
# Multi-head emission matrix
# ---------------------------------------------------------------------------

def build_emission(
    log_probs:    dict[str, np.ndarray],  # head → (T, C_head)
    phases:       list[tuple],            # (shape, att, hand_type, cloc, ctype, motion, nd_shape, nd_att)
    shape_to_idx: dict[str, int],
    att_to_idx:   dict[str, int],
    motion_to_idx:dict[str, int],
    cloc_to_idx:  dict[str, int],
    ctype_to_idx: dict[str, int],
    weights:      dict[str, float],
) -> np.ndarray:
    """
    Build (T, L) emission matrix where entry [t, l] is the combined log-prob
    that frame t belongs to shape segment l.
    """
    T = next(iter(log_probs.values())).shape[0]
    L = len(phases)
    em = np.zeros((T, L), dtype=np.float32)

    sign_idx = STATE_TO_IDX["sign"]

    for l, (shape, att, hand_type, cloc, ctype, motion, nd_shape, nd_att) in enumerate(phases):
        # State: frame should be in "sign" state
        if "state" in log_probs and weights.get("state", 0) > 0:
            em[:, l] += weights["state"] * log_probs["state"][:, sign_idx]

        # Shape
        if shape in shape_to_idx and weights.get("shape", 0) > 0:
            em[:, l] += weights["shape"] * log_probs["shape"][:, shape_to_idx[shape]]

        # Attitude (skip if unknown)
        if att is not None and att in att_to_idx and weights.get("att", 0) > 0:
            em[:, l] += weights["att"] * log_probs["att"][:, att_to_idx[att]]

        # Hand type: 0=one, 1=two
        if weights.get("hand_type", 0) > 0:
            ht_idx = 1 if hand_type == "two" else 0
            em[:, l] += weights["hand_type"] * log_probs["hand_type"][:, ht_idx]

        # Motion direction
        if motion is not None and motion in motion_to_idx and weights.get("motion", 0) > 0:
            em[:, l] += weights["motion"] * log_probs["motion"][:, motion_to_idx[motion]]

        # Contact location
        ci = cloc_to_idx.get(cloc or "none", 0)
        if weights.get("contact_loc", 0) > 0:
            em[:, l] += weights["contact_loc"] * log_probs["contact_loc"][:, ci]

        # Contact type
        ct = ctype_to_idx.get(ctype or "none", 0)
        if weights.get("contact_type", 0) > 0:
            em[:, l] += weights["contact_type"] * log_probs["contact_type"][:, ct]

    return em  # (T, L)
