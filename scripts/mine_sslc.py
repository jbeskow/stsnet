"""
Mine SSLC continuous signing corpus for gloss instances that match SSLL
dictionary signs by embedding distance (ClipClassifier clip_emb).

Steps:
  1. Load SSLL training clips → extract 256-dim clip_emb per clip.
     Group by gloss; keep all variants with their phonological targets.

  2. Iterate over SSLC gloss windows whose gloss appears in SSLL.
     Extract clip_emb for each window.

  3. For each SSLC instance, find the nearest SSLL variant (cosine distance);
     keep instances below --threshold and transfer the matched targets.

  4. Write sslc_mined.json.

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n slp python -u scripts/mine_sslc.py \\
        --clip_ckpt runs/clip_v02/best.pt \\
        --threshold 0.4 \\
        --out runs/sslc_mined.json
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from stsnet.data.pose_io import load_pose_streams, set_pose_cache_dir
from stsnet.data.ssll_clip import SSLLClipDataset
from stsnet.clip_classifier import ClipClassifier


SSLC_CSV      = "/nfs/signbot1/data/SSLC/sslc_glosses.csv"
SSLC_POSE_DIR = Path("/nfs/signbot1/data/SSLC/pose")
SSLL_CSV      = "/nfs/signbot1/data/SSLL/sign_data_with_signer_fr.csv"
SSLL_POSE_DIR = Path("/nfs/signbot1/data/SSLL/pose")
VOCAB_FILE    = "data/sts_handformer.txt"
PSEUDO_JSON   = "pseudo_signing.json"
FPS           = 25

EXCLUDED_SIGNERS = {
    "S003", "S017", "S023", "S025", "S037",            # left-handed
    "S015", "S021", "S029", "S031", "S035", "S041",    # unclear
}


# ── Model loading ─────────────────────────────────────────────────────────────

def load_clip_model(ckpt_path: str, device):
    model, _ = ClipClassifier.from_checkpoint(ckpt_path, map_location="cpu")
    return model.eval().to(device)


# ── Embedding helpers ──────────────────────────────────────────────────────────

def get_clip_emb(model, pose_path: Path, sign_start: int, sign_end: int,
                 device) -> np.ndarray | None:
    """Return (D,) clip_emb for a single SSLL sign window."""
    streams = load_pose_streams(pose_path, "right", mirror_left=True)
    if streams is None:
        return None
    n_dims = getattr(model, "n_dims", 3)
    if n_dims < 3:
        streams = {k: v[..., :n_dims] for k, v in streams.items()}

    def t(a):
        return torch.from_numpy(a).unsqueeze(0).to(device)

    T = streams["dominant"].shape[0]
    with torch.no_grad():
        out = model(
            t(streams["dominant"]), t(streams["nondominant"]),
            t(streams["body"]),     t(streams["face"]),
            sign_start=torch.tensor([sign_start], dtype=torch.long, device=device),
            sign_end=  torch.tensor([sign_end],   dtype=torch.long, device=device),
            lengths=   torch.tensor([T],          dtype=torch.long, device=device),
        )
    return out["clip_emb"].squeeze(0).cpu().numpy()


def get_clip_emb_slice(model, pose_path: Path, f0: int, f1: int,
                       device) -> np.ndarray | None:
    """Return (D,) clip_emb for a SSLC gloss window (entire slice = sign)."""
    streams = load_pose_streams(pose_path, "right", mirror_left=True)
    if streams is None:
        return None
    streams = {k: v[f0:f1] for k, v in streams.items()}
    n_dims = getattr(model, "n_dims", 3)
    if n_dims < 3:
        streams = {k: v[..., :n_dims] for k, v in streams.items()}
    T = streams["dominant"].shape[0]
    if T == 0:
        return None

    def t(a):
        return torch.from_numpy(a).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(
            t(streams["dominant"]), t(streams["nondominant"]),
            t(streams["body"]),     t(streams["face"]),
            sign_start=torch.zeros(1, dtype=torch.long, device=device),
            sign_end=  torch.tensor([T], dtype=torch.long, device=device),
            lengths=   torch.tensor([T], dtype=torch.long, device=device),
        )
    return out["clip_emb"].squeeze(0).cpu().numpy()


# ── SSLL reference embeddings ─────────────────────────────────────────────────

def ssll_train_indices(dataset: SSLLClipDataset, val_frac=0.15, seed=42):
    signers = sorted({s["signer"] for s in dataset.samples})
    rng = random.Random(seed)
    rng.shuffle(signers)
    n_val   = max(1, round(len(signers) * val_frac))
    val_set = set(signers[:n_val])
    return [i for i, s in enumerate(dataset.samples) if s["signer"] not in val_set]


def sample_targets(s: dict) -> dict:
    return {
        "shape_target":        s["shape_target"].tolist(),
        "nondom_shape_target": s["nondom_shape_target"].tolist(),
        "att_target":          s["att_target"].tolist(),
        "cloc_target":         s["cloc_target"].tolist(),
        "ctype_target":        s["ctype_target"].tolist(),
        "motion_target":       s["motion_target"].tolist(),
        "hand_type":           int(s["hand_type"]),
    }


def build_ssll_refs(model, dataset: SSLLClipDataset, train_idx, device):
    ssll_embs    = defaultdict(list)
    ssll_targets = defaultdict(list)
    n = len(train_idx)
    for ii, idx in enumerate(train_idx):
        s = dataset.samples[idx]
        word = s["word"].strip().upper()
        if not word:
            continue
        if ii % 500 == 0:
            print(f"  SSLL [{ii}/{n}]...")
        emb = get_clip_emb(model, s["pose_path"], s["sign_start"], s["sign_end"], device)
        if emb is None:
            continue
        ssll_embs[word].append(emb)
        ssll_targets[word].append(sample_targets(s))

    result_embs    = {g: np.stack(v) for g, v in ssll_embs.items()}
    result_targets = dict(ssll_targets)
    print(f"  SSLL: {len(result_embs)} glosses, "
          f"{sum(len(v) for v in result_embs.values())} variants")
    return result_embs, result_targets


# ── SSLC gloss embeddings ─────────────────────────────────────────────────────

def build_sslc_embs(model, ssll_gloss_set: set, device,
                    sslc_train_signers: set | None = None):
    rows = []
    seen = set()
    with open(SSLC_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            signer = row.get("signer", "")
            if sslc_train_signers is not None:
                if signer not in sslc_train_signers:
                    continue
            else:
                if signer in EXCLUDED_SIGNERS:
                    continue
            gloss = row["gloss"].split("@")[0].strip().upper()
            if gloss not in ssll_gloss_set:
                continue
            key = (row["video_file"], row["start_time"], row["end_time"], gloss)
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "gloss":      gloss,
                "video_file": row["video_file"],
                "start_ms":   int(row["start_time"]),
                "end_ms":     int(row["end_time"]),
                "signer":     signer,
            })

    print(f"  SSLC: {len(rows)} candidates across "
          f"{len({r['video_file'] for r in rows})} videos")

    video_to_rows: dict[str, list] = defaultdict(list)
    for r in rows:
        video_to_rows[r["video_file"]].append(r)

    results = []
    n_videos = len(video_to_rows)
    for vi, (video_file, occ_rows) in enumerate(video_to_rows.items()):
        if vi % 50 == 0:
            print(f"  SSLC video [{vi+1}/{n_videos}] {video_file}")
        pose_path = SSLC_POSE_DIR / (video_file + ".pose")
        if not pose_path.exists():
            continue
        for r in occ_rows:
            f0  = max(0, int(r["start_ms"] / 1000.0 * FPS))
            f1  = max(f0 + 1, int(r["end_ms"] / 1000.0 * FPS))
            emb = get_clip_emb_slice(model, pose_path, f0, f1, device)
            if emb is None:
                continue
            results.append({**r, "embedding": emb})

    print(f"  SSLC: {len(results)} instances with embeddings")
    return results


# ── Match and filter ──────────────────────────────────────────────────────────

def cosine_dist(query: np.ndarray, keys: np.ndarray) -> np.ndarray:
    q = query / (np.linalg.norm(query) + 1e-8)
    k = keys  / (np.linalg.norm(keys, axis=1, keepdims=True) + 1e-8)
    return 1.0 - (k @ q)


def match_and_filter(sslc_instances, ssll_embs, ssll_targets, threshold):
    mined = []
    by_gloss: dict[str, list] = defaultdict(list)
    for inst in sslc_instances:
        by_gloss[inst["gloss"]].append(inst)

    for gloss, instances in by_gloss.items():
        ref_embs    = ssll_embs[gloss]
        ref_targets = ssll_targets[gloss]
        for inst in instances:
            dists    = cosine_dist(inst["embedding"], ref_embs)
            best_idx = int(np.argmin(dists))
            dist     = float(dists[best_idx])
            if dist > threshold:
                continue
            mined.append({
                "video_file": inst["video_file"],
                "start_ms":   inst["start_ms"],
                "end_ms":     inst["end_ms"],
                "gloss":      gloss,
                "signer":     inst["signer"],
                "dist":       round(dist, 4),
                **ref_targets[best_idx],
            })

    return mined


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip_ckpt",       required=True,
                    help="ClipClassifier checkpoint (best.pt from train_clip.py)")
    ap.add_argument("--pose_cache",      default="/nfs/signbot1/data/SSLL/pose_cache")
    ap.add_argument("--sslc_pose_cache", default="/nfs/signbot1/data/SSLC/pose_cache")
    ap.add_argument("--threshold",       type=float, default=0.4)
    ap.add_argument("--val_frac",        type=float, default=0.15)
    ap.add_argument("--seed",            type=int,   default=42)
    ap.add_argument("--out",             default="runs/sslc_mined.json")
    ap.add_argument("--embs_cache",      default=None,
                    help="Save/load intermediate embeddings to skip GPU on re-runs")
    ap.add_argument("--sslc_split_json", default=None,
                    help="JSON with {train: [signers]}; restricts SSLC to train signers")
    ap.add_argument("--device",          default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_pose_cache_dir(args.pose_cache)

    print(f"Loading ClipClassifier from {args.clip_ckpt}...")
    model = load_clip_model(args.clip_ckpt, device)

    print("Loading SSLL dataset...")
    ds = SSLLClipDataset(
        csv_path    = SSLL_CSV,
        pose_dir    = SSLL_POSE_DIR,
        vocab_file  = VOCAB_FILE,
        pseudo_json = PSEUDO_JSON,
        noise_std   = 0.0,
    )
    ds.eval()
    train_idx = ssll_train_indices(ds, args.val_frac, args.seed)
    print(f"  Using {len(train_idx)} SSLL train clips")

    cache_path = Path(args.embs_cache) if args.embs_cache else None
    if cache_path and cache_path.exists():
        print(f"Loading cached embeddings from {cache_path}...")
        cache          = np.load(cache_path, allow_pickle=True)
        ssll_embs      = cache["ssll_embs"].item()
        ssll_targets   = cache["ssll_targets"].item()
        sslc_instances = list(cache["sslc_instances"])
        print(f"  SSLL: {len(ssll_embs)} glosses  |  SSLC: {len(sslc_instances)} instances")
    else:
        print("\nStep 1: Extracting SSLL reference embeddings...")
        ssll_embs, ssll_targets = build_ssll_refs(model, ds, train_idx, device)

        print("\nStep 2: Extracting SSLC gloss embeddings...")
        if args.sslc_pose_cache:
            set_pose_cache_dir(args.sslc_pose_cache)
        sslc_train_signers = None
        if args.sslc_split_json:
            with open(args.sslc_split_json) as f:
                sslc_train_signers = set(json.load(f)["train"])
            print(f"  Restricting SSLC to train signers: {sorted(sslc_train_signers)}")
        sslc_instances = build_sslc_embs(model, set(ssll_embs.keys()), device,
                                         sslc_train_signers)

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path,
                     ssll_embs=ssll_embs, ssll_targets=ssll_targets,
                     sslc_instances=np.array(sslc_instances, dtype=object))
            print(f"  Cached to {cache_path}")

    print(f"\nStep 3: Matching (threshold={args.threshold})...")
    mined = match_and_filter(sslc_instances, ssll_embs, ssll_targets, args.threshold)

    by_gloss = defaultdict(int)
    for m in mined:
        by_gloss[m["gloss"]] += 1
    print(f"\nSelected {len(mined)} instances from {len(by_gloss)} glosses")
    for g, n in sorted(by_gloss.items(), key=lambda x: -x[1])[:20]:
        print(f"  {g:<24} {n}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(mined, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
