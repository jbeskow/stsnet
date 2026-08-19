"""
Generate a cold-start seed alignment CSV.

Uses pseudo_signing.json for per-clip rest/prep/retract boundaries
(no trained model required) and splits the inner signing window equally
across the shape phases parsed from description text.

Structure per clip:
  rest        : 0 → sign_start
  __prep__    : sign_start → prep_end          (fixed fraction of window)
  shape_0     : prep_end → ...                 (equal split)
  ...
  shape_N     : ... → retract_start
  __retract__ : retract_start → sign_end       (fixed duration)
  rest        : sign_end → full_len

Usage:
    python scripts/make_seed_alignment.py \\
        --pseudo_signing pseudo_signing.json \\
        --output checkpoints/ctc/align_seed.csv
"""

import argparse
import csv
import json
import os
from pathlib import Path

from stsnet.viterbi import HEURISTIC_PARAMS, PREP_LABEL, RETRACT_LABEL
from stsnet.data.description import (
    load_handshape_vocab,
    _parse_description,
    load_llm_parse_cache,
)


def make_seed(args):
    with open(args.pseudo_signing) as f:
        pseudo = json.load(f)

    shapes_vocab = set(load_handshape_vocab(args.vocab_file))
    shape_fuzzy  = {v.lower().replace(" ", "").replace("-", ""): v for v in shapes_vocab}

    def match(raw):
        if raw in shapes_vocab: return raw
        return shape_fuzzy.get(raw.lower().replace(" ", "").replace("-", ""))

    if args.llm_cache:
        load_llm_parse_cache(args.llm_cache)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    written = skipped = 0

    with open(args.csv_path, newline="", encoding="utf-8") as cf, \
         open(args.output,   "w", newline="", encoding="utf-8") as of:

        reader = csv.DictReader(cf)
        writer = csv.writer(of)
        writer.writerow(["pose_file", "gloss", "signer", "handedness",
                         "hand", "label", "start_frame", "end_frame"])

        for row in reader:
            movie = (row.get("movie") or "").strip()
            if not movie or movie == "nan":
                skipped += 1; continue
            pose_fname = os.path.basename(movie) + ".pose"

            if pose_fname not in pseudo:
                skipped += 1; continue
            p          = pseudo[pose_fname]
            sign_start = p["sign_start"]
            sign_end   = p["sign_end"]
            full_len   = p["T"]

            if sign_end <= sign_start:
                skipped += 1; continue

            phases_raw = _parse_description(row.get("description", ""))
            shapes = []
            for tup in phases_raw:
                c = match(tup[0])
                if c: shapes.append(c)

            if not shapes:
                skipped += 1; continue
            if args.max_shapes and len(shapes) > args.max_shapes:
                skipped += 1; continue

            W   = sign_end - sign_start
            mid = sign_start + W // 2

            prep_end      = sign_start + max(2, round(W * HEURISTIC_PARAMS["prep_frac"]))
            prep_end      = min(prep_end, mid)
            retract_start = sign_end - max(2, HEURISTIC_PARAMS["retract_dur"])
            retract_start = max(retract_start, mid)

            inner_len = retract_start - prep_end
            N = len(shapes)
            if inner_len < N:
                skipped += 1; continue

            seg_w = inner_len // N
            segs  = []
            if sign_start > 0:
                segs.append(("rest", 0, sign_start))
            if prep_end > sign_start:
                segs.append((PREP_LABEL, sign_start, prep_end))
            cur = prep_end
            for k, sh in enumerate(shapes):
                nxt = cur + seg_w if k < N - 1 else retract_start
                segs.append((sh, cur, nxt))
                cur = nxt
            if retract_start < sign_end:
                segs.append((RETRACT_LABEL, retract_start, sign_end))
            if sign_end < full_len:
                segs.append(("rest", sign_end, full_len))

            for lbl, sf, ef in segs:
                writer.writerow([pose_fname, "", row.get("signer", ""),
                                 "right", "dominant", lbl, sf, ef])
            written += 1

    print(f"Seed alignment: {written} clips written, {skipped} skipped → {args.output}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pseudo_signing", default="pseudo_signing.json")
    ap.add_argument("--output",  default="checkpoints/ctc/align_seed.csv")
    ap.add_argument("--csv_path", default="/nfs/signbot1/data/SSLL/sign_data_with_signer_fr.csv")
    ap.add_argument("--vocab_file", default="/nfs/signbot1/data/SSLL/sts_handformer.txt")
    ap.add_argument("--llm_cache", default=None)
    ap.add_argument("--max_shapes", type=int, default=3)
    args = ap.parse_args()
    make_seed(args)


if __name__ == "__main__":
    main()
