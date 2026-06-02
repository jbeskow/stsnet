"""
Dataset wrapping sslc_mined.json — SSLC gloss clips with phonological labels
transferred from the nearest-matching SSLL dictionary variant (see scripts/mine_sslc.py).

Compatible with SSLLClipDataset.__getitem__ output; collate_clip works unchanged.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from stsnet.data.pose_io import load_pose_streams, set_pose_cache_dir
from stsnet.data.ssll_clip import time_stretch_streams, jitter_sign_window


class SSLCMinedDataset(Dataset):
    """
    SSLC gloss clips selected by ClipClassifier embedding proximity to SSLL dictionary.

    Args:
        json_path:          path to sslc_mined.json produced by scripts/mine_sslc.py
        pose_dir:           directory containing SSLC .pose files
        noise_std:          Gaussian pose augmentation (training only)
        pose_cache_dir:     pre-built .npz pose cache (speeds up loading)
        n_coords:           coordinate dims (3 = xyz, 2 = xy)
        time_stretch_range: (min, max) stretch factors for training augmentation
        temporal_jitter:    max frame shift for sign boundary jitter (0 = disabled)
    """

    def __init__(
        self,
        json_path:          str | Path,
        pose_dir:           str | Path,
        noise_std:          float = 0.0,
        pose_cache_dir:     str | Path | None = None,
        n_coords:           int = 3,
        time_stretch_range: tuple[float, float] = (1.0, 1.0),
        temporal_jitter:    int = 0,
    ):
        self.pose_dir           = Path(pose_dir)
        self.noise_std          = noise_std
        self._training          = True
        self.n_coords           = n_coords
        self.time_stretch_range = time_stretch_range
        self.temporal_jitter    = temporal_jitter

        if pose_cache_dir is not None:
            set_pose_cache_dir(pose_cache_dir)

        with open(json_path, encoding="utf-8") as f:
            raw = json.load(f)

        self.samples = []
        for r in raw:
            pose_path = self.pose_dir / (r["video_file"] + ".pose")
            if not pose_path.exists():
                continue
            n_shapes = len(r["shape_target"])
            self.samples.append({
                "pose_path":           pose_path,
                "start_ms":            r["start_ms"],
                "end_ms":              r["end_ms"],
                "gloss":               r["gloss"],
                "signer":              r.get("signer", ""),
                "dist":                r["dist"],
                "shape_target":        torch.tensor(r["shape_target"],  dtype=torch.float),
                "nondom_shape_target": torch.tensor(
                    r.get("nondom_shape_target", [0.0] * n_shapes), dtype=torch.float),
                "att_target":          torch.tensor(r["att_target"],    dtype=torch.float),
                "nondom_att_target":   torch.tensor(
                    r.get("nondom_att_target", [0.0] * len(r["att_target"])), dtype=torch.float),
                "cloc_target":         torch.tensor(r["cloc_target"],   dtype=torch.float),
                "ctype_target":        torch.tensor(r["ctype_target"],  dtype=torch.float),
                "motion_target":       torch.tensor(r["motion_target"], dtype=torch.float),
                "hand_type":           torch.tensor(r["hand_type"],     dtype=torch.long),
            })

        print(f"SSLCMinedDataset: {len(self.samples)} samples "
              f"({len(raw) - len(self.samples)} pose files missing)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]

        streams = load_pose_streams(s["pose_path"], "right", mirror_left=True)
        if streams is None:
            streams = {k: np.zeros((1, j, 3), dtype=np.float32)
                       for k, j in [("dominant", 21), ("nondominant", 21),
                                    ("body", 12), ("face", 25)]}

        fps = 25
        f0 = max(0, int(s["start_ms"] / 1000.0 * fps))
        f1 = max(f0 + 1, int(s["end_ms"] / 1000.0 * fps))

        if self._training and self.temporal_jitter > 0:
            T_full = next(iter(streams.values())).shape[0]
            f0, f1 = jitter_sign_window(f0, f1, T_full, self.temporal_jitter)

        streams = {k: v[f0:f1] for k, v in streams.items()}

        if self._training:
            lo, hi = self.time_stretch_range
            if lo < hi:
                streams = time_stretch_streams(streams, random.uniform(lo, hi))

        item = {k: torch.from_numpy(v) for k, v in streams.items()}

        if self.n_coords < 3:
            for k in ("dominant", "nondominant", "body", "face"):
                if k in item:
                    item[k] = item[k][..., :self.n_coords]

        T = item["dominant"].shape[0]
        item["sign_start"] = torch.tensor(0, dtype=torch.long)
        item["sign_end"]   = torch.tensor(T, dtype=torch.long)
        item["signer"]     = s["signer"]
        item["word"]       = s["gloss"]

        for key in ("shape_target", "nondom_shape_target", "att_target",
                    "nondom_att_target", "cloc_target", "ctype_target",
                    "motion_target", "hand_type"):
            item[key] = s[key]

        if self.noise_std > 0.0 and self._training:
            for k in ("dominant", "nondominant", "body"):
                if k in item:
                    item[k] = item[k] + torch.randn_like(item[k]) * self.noise_std

        return item

    def train(self, mode=True):
        self._training = mode
        return self

    def eval(self):
        return self.train(False)
