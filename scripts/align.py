"""
Viterbi forced alignment using STSNet multi-head emissions.

For each clip the emission score for frame t being in shape segment l is:

    emission(t, l) = Σ_h  w_h · log P_h(expected_label_l | x_t)

where heads h ∈ {state, shape, att, hand_type, motion, contact_loc, contact_type}.

Outer prep/retract boundaries are set by heuristic; Viterbi runs on inner shapes.

Output CSV format:
    pose_file, gloss, signer, handedness, hand, label, start_frame, end_frame

Usage:
    conda run -n slp --no-capture-output python -u scripts/align.py \\
        --ckpt checkpoints/stsnet_v1/best.pt \\
        --pseudo_signing pseudo_signing.json \\
        --output checkpoints/ctc/align_sts_v1.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from stsnet.data.pose_io import set_pose_cache_dir
from stsnet.data.description import load_llm_parse_cache, MOTION_DIRECTIONS
from stsnet.data.contact import CONTACT_LOCATIONS, CONTACT_TYPES
from stsnet.data.multihead import STATE_TO_IDX
from stsnet.data.align_dataset import (
    STSAlignDataset,
    collate_align,
    build_emission,
)
from stsnet.model import STSNet
from stsnet.viterbi import (
    ctc_forced_align,
    frame_labels_to_inner_segs,
    equal_spacing_inner,
    HEURISTIC_PARAMS,
    PREP_LABEL,
    RETRACT_LABEL,
)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.cache_dir:
        set_pose_cache_dir(args.cache_dir)
    if args.llm_cache:
        load_llm_parse_cache(args.llm_cache)

    # Load checkpoint and build model
    ck = torch.load(args.ckpt, map_location="cpu")
    ck_args = ck.get("args", {})
    vocabs   = ck.get("vocabs", {})

    shape_to_idx  = vocabs.get("shape_to_idx",  {})
    att_to_idx    = vocabs.get("att_to_idx",     {})
    state_to_idx  = vocabs.get("state_to_idx",   STATE_TO_IDX)
    motion_to_idx = vocabs.get("motion_to_idx",  {d: i for i, d in enumerate(MOTION_DIRECTIONS)})
    cloc_to_idx   = vocabs.get("cloc_to_idx",    {l: i for i, l in enumerate(CONTACT_LOCATIONS)})
    ctype_to_idx  = vocabs.get("ctype_to_idx",   {t: i for i, t in enumerate(CONTACT_TYPES)})

    num_shapes = len(shape_to_idx)
    num_atts   = len(att_to_idx)
    num_cloc   = len(cloc_to_idx)
    num_ctype  = len(ctype_to_idx)

    streams = tuple(ck_args.get("streams", ["dom", "nondom", "body", "face"]))
    model = STSNet(
        num_shapes        = num_shapes,
        num_atts          = num_atts,
        num_contact_locs  = num_cloc,
        num_contact_types = num_ctype,
        num_nondom_shapes = num_shapes,
        num_nondom_atts   = num_atts,
        hidden_dim        = ck_args.get("hidden_dim", 512),
        conv_layers       = ck_args.get("conv_layers", 3),
        kernel_size       = ck_args.get("kernel_size", 5),
        dropout           = 0.0,
        bilstm_layers     = ck_args.get("bilstm_layers", 1),
        streams           = streams,
    ).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    print(f"Loaded {args.ckpt}  streams={streams}")

    weights = {
        "state":        args.w_state,
        "shape":        args.w_shape,
        "att":          args.w_att,
        "hand_type":    args.w_hand_type,
        "motion":       args.w_motion,
        "contact_loc":  args.w_contact_loc,
        "contact_type": args.w_contact_type,
    }
    print("Weights:", {k: v for k, v in weights.items() if v > 0})

    ds = STSAlignDataset(
        csv_path       = args.csv_path,
        pose_dir       = args.pose_dir,
        vocab_file     = args.vocab_file,
        alignment_csv  = args.alignment_csv,
        pseudo_signing = args.pseudo_signing,
        llm_cache      = args.llm_cache,
        max_shapes     = args.max_shapes,
    )

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_align)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    done = fallback = 0

    with open(args.output, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(["pose_file", "gloss", "signer", "handedness",
                         "hand", "label", "start_frame", "end_frame"])

        for batch in loader:
            B = batch["dominant"].shape[0]
            dom    = batch["dominant"].to(device)
            nondom = batch["nondominant"].to(device)
            body   = batch["body"].to(device)
            face   = batch["face"].to(device)
            lens   = batch["lengths"].to(device)

            with torch.no_grad():
                out = model(dom, nondom, body, face, lengths=lens)

            head_map = {
                "state":        "state_logits",
                "shape":        "shape_logits",
                "att":          "att_logits",
                "hand_type":    "hand_type_logits",
                "motion":       "motion_logits",
                "contact_loc":  "contact_loc_logits",
                "contact_type": "contact_type_logits",
            }
            log_probs_batch = {
                h: F.log_softmax(out[lk], dim=-1).cpu().numpy()
                for h, lk in head_map.items() if lk in out
            }

            for i in range(B):
                pose_fname  = batch["pose_fname"][i]
                signer      = batch["signer"][i]
                sign_start  = batch["sign_start"][i]
                sign_end    = batch["sign_end"][i]
                full_len    = batch["full_len"][i]
                phases      = batch["phases"][i]
                shapes_list = batch["shapes_list"][i]
                T_i         = int(lens[i].item())

                lp_i = {h: lp[i, :T_i] for h, lp in log_probs_batch.items()}

                # Heuristic outer bounds
                W  = sign_end - sign_start
                mid = sign_start + W // 2
                prep_end      = sign_start + max(2, round(W * HEURISTIC_PARAMS["prep_frac"]))
                prep_end      = min(prep_end, mid)
                retract_start = sign_end - max(2, HEURISTIC_PARAMS["retract_dur"])
                retract_start = max(retract_start, mid)

                inner_s   = prep_end      - sign_start
                inner_e   = retract_start - sign_start
                inner_len = inner_e - inner_s
                L         = len(phases)

                if L == 0 or inner_len < L * args.min_dur:
                    inner_segs = equal_spacing_inner(shapes_list, prep_end, retract_start)
                    fallback  += 1
                else:
                    lp_inner = {h: lp[inner_s:inner_e] for h, lp in lp_i.items()}
                    em = build_emission(lp_inner, phases,
                                        shape_to_idx, att_to_idx, motion_to_idx,
                                        cloc_to_idx, ctype_to_idx, weights)
                    frame_labels = ctc_forced_align(em, list(range(L)),
                                                    blank=0, min_dur=args.min_dur)
                    inner_segs = frame_labels_to_inner_segs(
                        frame_labels, shapes_list, prep_end)
                    done += 1

                # Assemble and write
                segs = []
                if sign_start > 0:
                    segs.append(("rest", 0, sign_start))
                if prep_end > sign_start:
                    segs.append((PREP_LABEL, sign_start, prep_end))
                segs.extend(inner_segs)
                if retract_start < sign_end:
                    segs.append((RETRACT_LABEL, retract_start, sign_end))
                if sign_end < full_len:
                    segs.append(("rest", sign_end, full_len))

                for lbl, sf, ef in segs:
                    writer.writerow([pose_fname, "", signer, "right",
                                     "dominant", lbl, sf, ef])

    print(f"\nDone.  Viterbi={done}  fallback={fallback}")
    print(f"Output: {args.output}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",         required=True)
    ap.add_argument("--output",       required=True)
    ap.add_argument("--pseudo_signing", default="pseudo_signing.json")
    ap.add_argument("--alignment_csv",  default=None)
    ap.add_argument("--csv_path",  default="/nfs/signbot1/data/SSLL/sign_data_with_signer_fr.csv")
    ap.add_argument("--pose_dir",  default="/nfs/signbot1/data/SSLL/pose")
    ap.add_argument("--cache_dir", default="/nfs/signbot1/data/SSLL/pose_cache")
    ap.add_argument("--vocab_file",default="/nfs/signbot1/data/SSLL/sts_handformer.txt")
    ap.add_argument("--llm_cache", default=None)
    ap.add_argument("--max_shapes",type=int,   default=3)
    ap.add_argument("--min_dur",   type=int,   default=3)
    ap.add_argument("--batch_size",type=int,   default=64)
    ap.add_argument("--num_workers",type=int,  default=0)
    ap.add_argument("--w_state",      type=float, default=1.0)
    ap.add_argument("--w_shape",      type=float, default=1.0)
    ap.add_argument("--w_att",        type=float, default=0.7)
    ap.add_argument("--w_hand_type",  type=float, default=0.3)
    ap.add_argument("--w_motion",     type=float, default=0.5)
    ap.add_argument("--w_contact_loc",type=float, default=0.0)
    ap.add_argument("--w_contact_type",type=float,default=0.0)
    return ap.parse_args()


if __name__ == "__main__":
    main(parse_args())
