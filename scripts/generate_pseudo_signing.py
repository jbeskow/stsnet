"""
Compute pseudo signing labels from kinematics alone (no alignment CSV).

For each clip, uses the dominant-hand wrist trajectory (from the body stream,
shoulder-normalised) to estimate which frames are "signing" vs "rest", based on:
  - wrist velocity  (primary: hand starts moving → signing)
  - distance from rest position  (secondary: hand lifted away from neutral)
  - time-position prior: first/last REST_PAD frames are forced to rest

Output JSON: { pose_file → {"sign_start": int, "sign_end": int, "T": int} }

Usage:
    conda run -n slp python scripts/generate_pseudo_signing.py \\
        --out pseudo_signing.json
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

# Body stream layout (MediaPipe pose indices 11-22, 0-indexed in the body array)
_BODY_L_WRIST = 4   # MP landmark 15
_BODY_R_WRIST = 5   # MP landmark 16


def _load_wrist(cache_dir: Path, pose_fname: str, handedness: str) -> np.ndarray | None:
    """
    Load dominant wrist trajectory from .npz cache.
    Returns (T, 3) shoulder-normalised wrist positions, or None on failure.
    Body joints: 0=L-shoulder, 1=R-shoulder, 4=L-wrist, 5=R-wrist.
    """
    npz_path = cache_dir / (pose_fname + ".npz")
    if not npz_path.exists():
        return None
    try:
        npz  = np.load(npz_path)
        body = npz["body"]          # (T, 12, 3)
    except Exception:
        return None

    joint = _BODY_R_WRIST if handedness == "right" else _BODY_L_WRIST
    wrist = body[:, joint, :]       # (T, 3)
    return wrist.astype(np.float32)


def _pseudo_sign_range(
    wrist:       np.ndarray,
    vel_thresh:  float,
    pos_thresh:  float,
    smooth_win:  int,
    rest_pad:    int,
    rest_ref:    int,
) -> tuple[int, int] | None:
    """
    Compute (sign_start, sign_end) from a (T, 3) wrist trajectory.
    Returns None if the clip is too short or has too many missing frames.
    """
    T = len(wrist)
    if T < 2 * rest_pad + 2:
        return None

    valid = ~np.any(np.isnan(wrist), axis=1)    # (T,)
    if valid.sum() < rest_pad + 1:
        return None

    # rest-position estimate from clip margins
    n = min(rest_ref, T // 4)
    margin = np.concatenate([wrist[:n], wrist[T - n:]], axis=0)
    rest_pos = np.nanmean(margin, axis=0)       # (3,)

    # velocity
    diff = np.diff(wrist, axis=0)               # (T-1, 3)
    vel  = np.sqrt(np.nansum(diff ** 2, axis=1))  # (T-1,)
    vel  = np.concatenate([[vel[0]], vel])       # (T,)  pad first frame

    # distance from rest position
    pos_dist = np.sqrt(np.nansum((wrist - rest_pos) ** 2, axis=1))  # (T,)

    # smooth both signals
    k = np.ones(smooth_win) / smooth_win
    vel_s  = np.convolve(np.nan_to_num(vel),      k, mode="same")
    dist_s = np.convolve(np.nan_to_num(pos_dist), k, mode="same")

    # signing score: above either threshold counts
    signing = (vel_s >= vel_thresh) | (dist_s >= pos_thresh)

    # force margins to rest regardless of score
    signing[:rest_pad]  = False
    signing[-rest_pad:] = False

    idxs = np.where(signing)[0]
    if len(idxs) == 0:
        # nothing detected — return mid-clip as a fallback
        return (rest_pad, T - rest_pad)

    return (int(idxs[0]), int(idxs[-1]) + 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path",   default="/nfs/signbot1/data/SSLL/sign_data_with_signer_fr.csv")
    parser.add_argument("--cache_dir",  default="/nfs/signbot1/data/SSLL/pose_cache")
    parser.add_argument("--out",        default="pseudo_signing.json")
    parser.add_argument("--vel_thresh", type=float, default=0.04)
    parser.add_argument("--pos_thresh", type=float, default=0.12)
    parser.add_argument("--smooth_win", type=int,   default=5)
    parser.add_argument("--rest_pad",   type=int,   default=2)
    parser.add_argument("--rest_ref",   type=int,   default=3)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)

    clips: dict[str, str] = {}   # pose_fname → handedness
    with open(args.csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            movie = (row.get("movie") or "").strip()
            if not movie or movie == "nan":
                continue
            fname      = Path(movie).name
            pose_fname = fname + ".pose"
            hand       = (row.get("hand") or "right").strip().lower()
            clips[pose_fname] = hand

    print(f"Processing {len(clips)} clips...")

    results: dict[str, dict] = {}
    skipped = 0

    for pose_fname, handedness in clips.items():
        wrist = _load_wrist(cache_dir, pose_fname, handedness)
        if wrist is None:
            skipped += 1
            continue

        T   = len(wrist)
        out = _pseudo_sign_range(
            wrist,
            vel_thresh = args.vel_thresh,
            pos_thresh = args.pos_thresh,
            smooth_win = args.smooth_win,
            rest_pad   = args.rest_pad,
            rest_ref   = args.rest_ref,
        )
        if out is None:
            skipped += 1
            continue

        sign_start, sign_end = out
        results[pose_fname] = {
            "sign_start": sign_start,
            "sign_end":   sign_end,
            "T":          T,
        }

    with open(args.out, "w") as f:
        json.dump(results, f)

    print(f"Done. {len(results)} clips written to {args.out}  ({skipped} skipped).")


if __name__ == "__main__":
    main()
