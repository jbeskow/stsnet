"""
Train ClipClassifier (STS-Net v0.2): attention-pooled clip-level phonology model.

No alignment CSV required — uses pseudo_signing.json for sign-frame ranges and
description strings for phonological labels (including multi-phase signs).

Loss:
    BCE for shape, att, cloc, ctype, motion  (multi-hot targets)
    CE  for hand_type                         (single-label)

Usage:
    conda run -n slp python -u scripts/train_clip.py \\
        --out runs/clip_v02

Warm-start from v0.1 STSNet checkpoint:
    conda run -n slp python -u scripts/train_clip.py \\
        --out runs/clip_v02_warm \\
        --ckpt checkpoints/stsnet_base.pt

Resume / fine-tune from an existing ClipClassifier checkpoint:
    conda run -n slp python -u scripts/train_clip.py \\
        --out runs/clip_v02_ft \\
        --clip_ckpt runs/clip_v02/best.pt

Include mined SSLC data:
    conda run -n slp python -u scripts/train_clip.py \\
        --out runs/clip_v02_mined \\
        --mined_json runs/sslc_mined.json \\
        --dropout 0.3 --noise_std 0.02 --label_smoothing 0.1 \\
        --time_stretch_min 0.85 --time_stretch_max 1.15
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

from stsnet.data.pose_io import set_pose_cache_dir
from stsnet.data.ssll_clip import SSLLClipDataset, collate_clip
from stsnet.clip_classifier import ClipClassifier


# ── Loss helpers ──────────────────────────────────────────────────────────────

def bce_loss(logits: torch.Tensor, target: torch.Tensor,
             label_smoothing: float = 0.0) -> torch.Tensor:
    active = target.sum(dim=-1) > 0
    if active.sum() == 0:
        return logits.sum() * 0.0
    t = target[active]
    if label_smoothing > 0.0:
        t = t * (1.0 - label_smoothing) + label_smoothing / t.shape[-1]
    return F.binary_cross_entropy_with_logits(logits[active], t)


def ce_loss(logits: torch.Tensor, target: torch.Tensor,
            label_smoothing: float = 0.0) -> torch.Tensor:
    return F.cross_entropy(logits, target, ignore_index=-1,
                           label_smoothing=label_smoothing)


# ── Accuracy helpers ──────────────────────────────────────────────────────────

@torch.no_grad()
def bce_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    active = target.sum(dim=-1) > 0
    if active.sum() == 0:
        return float("nan")
    top1 = logits[active].argmax(dim=-1)
    hits = target[active].gather(1, top1.unsqueeze(1)).squeeze(1)
    return float(hits.float().mean())


@torch.no_grad()
def ce_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    valid = target != -1
    if valid.sum() == 0:
        return float("nan")
    return float((logits[valid].argmax(dim=-1) == target[valid]).float().mean())


# ── Train / val split ─────────────────────────────────────────────────────────

def signer_split(dataset: SSLLClipDataset, val_frac: float = 0.15, seed: int = 42):
    signers = sorted({s["signer"] for s in dataset.samples})
    rng = random.Random(seed)
    rng.shuffle(signers)
    n_val   = max(1, round(len(signers) * val_frac))
    val_set = set(signers[:n_val])
    train_idx = [i for i, s in enumerate(dataset.samples) if s["signer"] not in val_set]
    val_idx   = [i for i, s in enumerate(dataset.samples) if s["signer"] in val_set]
    print(f"Split: {len(train_idx)} train / {len(val_idx)} val "
          f"({len(signers) - n_val} train signers, {n_val} val signers)")
    return train_idx, val_idx


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",         default="/nfs/signbot1/data/SSLL/sign_data_with_signer_fr.csv")
    ap.add_argument("--pose_dir",    default="/nfs/signbot1/data/SSLL/pose")
    ap.add_argument("--vocab",       default="data/sts_handformer.txt")
    ap.add_argument("--pseudo",      default="pseudo_signing.json")
    ap.add_argument("--out",         default="runs/clip_v02")
    ap.add_argument("--ckpt",        default=None, help="v0.1 STSNet checkpoint for warm-start")
    ap.add_argument("--clip_ckpt",   default=None, help="ClipClassifier checkpoint for warm-start")
    ap.add_argument("--epochs",      type=int,   default=60)
    ap.add_argument("--batch_size",  type=int,   default=32)
    ap.add_argument("--lr",          type=float, default=3e-4)
    ap.add_argument("--hidden_dim",  type=int,   default=256)
    ap.add_argument("--conv_layers", type=int,   default=3)
    ap.add_argument("--dropout",     type=float, default=0.2)
    ap.add_argument("--noise_std",          type=float, default=0.005)
    ap.add_argument("--label_smoothing",    type=float, default=0.0)
    ap.add_argument("--weight_decay",       type=float, default=1e-4)
    ap.add_argument("--patience",           type=int,   default=0,
                    help="Early stopping patience (0=disabled)")
    ap.add_argument("--time_stretch_min",   type=float, default=1.0)
    ap.add_argument("--time_stretch_max",   type=float, default=1.0)
    ap.add_argument("--temporal_jitter",    type=int,   default=0)
    ap.add_argument("--val_frac",    type=float, default=0.15)
    ap.add_argument("--num_workers", type=int,   default=4)
    ap.add_argument("--streams",     nargs="+",  default=["dom", "nondom", "body", "face"])
    ap.add_argument("--wilor_dir",    default=None)
    ap.add_argument("--dino_dir",     default=None)
    ap.add_argument("--sapiens_dir",  default=None)
    ap.add_argument("--no_z",             action="store_true",
                    help="Use 2D (xy) pose only — strip depth coordinate")
    ap.add_argument("--nondom_shape_head", action="store_true")
    ap.add_argument("--mined_json",   default=None,
                    help="Path to sslc_mined.json (extends training set)")
    ap.add_argument("--pose_cache",   default="/nfs/signbot1/data/SSLL/pose_cache")
    ap.add_argument("--sslc_pose_cache", default=None)
    ap.add_argument("--device",       default="cuda")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.pose_cache:
        set_pose_cache_dir(args.pose_cache)

    _log_file = open(out_dir / "train.log", "w", buffering=1)
    class _Tee:
        def __init__(self, *s): self.streams = s
        def write(self, s):
            for st in self.streams: st.write(s)
        def flush(self):
            for st in self.streams: st.flush()
    sys.stdout = sys.stderr = _Tee(sys.__stdout__, _log_file)

    n_coords     = 2 if args.no_z else 3
    stretch_range = (args.time_stretch_min, args.time_stretch_max)

    print("Loading dataset...")
    ds = SSLLClipDataset(
        csv_path           = args.csv,
        pose_dir           = args.pose_dir,
        vocab_file         = args.vocab,
        pseudo_json        = args.pseudo,
        noise_std          = args.noise_std,
        wilor_dir          = args.wilor_dir,
        dino_dir           = args.dino_dir,
        sapiens_dir        = args.sapiens_dir,
        n_coords           = n_coords,
        time_stretch_range = stretch_range,
        temporal_jitter    = args.temporal_jitter,
    )

    train_idx, val_idx = signer_split(ds, args.val_frac)
    train_ds = Subset(ds, train_idx)
    val_ds   = Subset(ds, val_idx)
    val_ds.dataset.eval()

    if args.mined_json:
        from stsnet.data.sslc_mined import SSLCMinedDataset
        mined_ds = SSLCMinedDataset(
            args.mined_json,
            pose_dir           = "/nfs/signbot1/data/SSLC/pose",
            noise_std          = args.noise_std,
            pose_cache_dir     = args.sslc_pose_cache,
            n_coords           = n_coords,
            time_stretch_range = stretch_range,
            temporal_jitter    = args.temporal_jitter,
        )
        train_ds = ConcatDataset([train_ds, mined_ds])
        print(f"Extended train set: {len(train_ds)} clips "
              f"(SSLL {len(train_idx)} + SSLC mined {len(mined_ds)})")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_clip,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_clip,
                              pin_memory=True)

    vocab_sizes = dict(
        num_shapes       = ds.num_shapes,
        num_atts         = ds.num_atts,
        num_cloc         = ds.num_cloc,
        num_ctype        = ds.num_ctype,
        num_motion       = ds.num_motion,
        hidden_dim       = args.hidden_dim,
        conv_layers      = args.conv_layers,
        dropout          = args.dropout,
        streams          = tuple(args.streams),
        n_body           = 17 if args.sapiens_dir else 12,
        n_face           = 68 if args.sapiens_dir else 25,
        n_dims           = 2 if args.no_z else 3,
        has_nondom_shape = args.nondom_shape_head,
    )

    model = ClipClassifier(**vocab_sizes)
    if args.clip_ckpt:
        print(f"Warm-starting from ClipClassifier checkpoint {args.clip_ckpt}")
        ck = torch.load(args.clip_ckpt, map_location="cpu")
        saved   = ck["model_state_dict"]
        current = model.state_dict()
        filtered = {k: v for k, v in saved.items()
                    if k in current and v.shape == current[k].shape}
        skipped = [k for k in saved if k not in filtered]
        model.load_state_dict(filtered, strict=False)
        if skipped:
            print(f"  Skipped (shape mismatch): {skipped}")
    elif args.ckpt:
        print(f"Warm-starting from v0.1 checkpoint {args.ckpt}")
        model = ClipClassifier.from_stsnet_checkpoint(args.ckpt, **vocab_sizes)

    model.to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    vocab_meta = {
        "shape_to_idx":  ds.shape_to_idx,
        "att_to_idx":    ds.att_to_idx,
        "cloc_to_idx":   ds.cloc_to_idx,
        "ctype_to_idx":  ds.ctype_to_idx,
        "motion_to_idx": ds.motion_to_idx,
        "model_kwargs":  {k: (list(v) if isinstance(v, tuple) else v)
                          for k, v in vocab_sizes.items()},
    }
    with open(out_dir / "vocab.json", "w") as f:
        json.dump(vocab_meta, f, ensure_ascii=False, indent=2)

    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                        eta_min=args.lr / 20)
    writer     = SummaryWriter(out_dir / "tb")
    best_val   = float("inf")
    no_improve = 0

    HEAD_WEIGHTS = {
        "shape": 1.0, "att": 1.0, "hand_type": 0.5,
        "cloc": 0.5, "ctype": 0.5, "motion": 0.5,
        "nondom_shape": 0.5,
    }

    for epoch in range(1, args.epochs + 1):
        model.train()
        ds.train()
        train_loss = 0.0

        for batch in train_loader:
            dom  = batch["dominant"].to(device)
            ndom = batch["nondominant"].to(device)
            body = batch["body"].to(device)
            face = batch["face"].to(device)
            lens = batch["lengths"].to(device)
            ss   = batch["sign_start"].to(device)
            se   = batch["sign_end"].to(device)
            extra = {k: batch[k].to(device)
                     for k in ("wilor_dom", "wilor_nondom",
                               "dom_norm", "nondom_norm", "dino") if k in batch}

            out = model(dom, ndom, body, face, ss, se, lens, **extra)
            ls  = args.label_smoothing

            loss = (
                HEAD_WEIGHTS["shape"]     * bce_loss(out["shape_logits"],     batch["shape_target"].to(device), ls)
              + HEAD_WEIGHTS["att"]       * bce_loss(out["att_logits"],       batch["att_target"].to(device),   ls)
              + HEAD_WEIGHTS["cloc"]      * bce_loss(out["cloc_logits"],      batch["cloc_target"].to(device),  ls)
              + HEAD_WEIGHTS["ctype"]     * bce_loss(out["ctype_logits"],     batch["ctype_target"].to(device), ls)
              + HEAD_WEIGHTS["motion"]    * bce_loss(out["motion_logits"],    batch["motion_target"].to(device),ls)
              + HEAD_WEIGHTS["hand_type"] * ce_loss(out["hand_type_logits"],  batch["hand_type"].to(device),    ls)
            )
            if args.nondom_shape_head:
                loss = loss + HEAD_WEIGHTS["nondom_shape"] * bce_loss(
                    out["nondom_shape_logits"], batch["nondom_shape_target"].to(device), ls)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        sched.step()

        model.eval()
        val_loss = shape_acc = att_acc = ht_acc = nondom_shape_acc = 0.0
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                dom  = batch["dominant"].to(device)
                ndom = batch["nondominant"].to(device)
                body = batch["body"].to(device)
                face = batch["face"].to(device)
                lens = batch["lengths"].to(device)
                ss   = batch["sign_start"].to(device)
                se   = batch["sign_end"].to(device)
                extra = {k: batch[k].to(device)
                         for k in ("wilor_dom", "wilor_nondom",
                               "dom_norm", "nondom_norm", "dino") if k in batch}

                out = model(dom, ndom, body, face, ss, se, lens, **extra)
                s_t = batch["shape_target"].to(device)
                a_t = batch["att_target"].to(device)
                ht  = batch["hand_type"].to(device)

                loss = (
                    HEAD_WEIGHTS["shape"]     * bce_loss(out["shape_logits"],  s_t)
                  + HEAD_WEIGHTS["att"]       * bce_loss(out["att_logits"],    a_t)
                  + HEAD_WEIGHTS["cloc"]      * bce_loss(out["cloc_logits"],   batch["cloc_target"].to(device))
                  + HEAD_WEIGHTS["ctype"]     * bce_loss(out["ctype_logits"],  batch["ctype_target"].to(device))
                  + HEAD_WEIGHTS["motion"]    * bce_loss(out["motion_logits"], batch["motion_target"].to(device))
                  + HEAD_WEIGHTS["hand_type"] * ce_loss(out["hand_type_logits"], ht)
                )
                if args.nondom_shape_head:
                    ns_t = batch["nondom_shape_target"].to(device)
                    loss = loss + HEAD_WEIGHTS["nondom_shape"] * bce_loss(
                        out["nondom_shape_logits"], ns_t)
                    nondom_shape_acc += bce_accuracy(out["nondom_shape_logits"], ns_t)

                val_loss  += loss.item()
                shape_acc += bce_accuracy(out["shape_logits"], s_t)
                att_acc   += bce_accuracy(out["att_logits"],   a_t)
                ht_acc    += ce_accuracy(out["hand_type_logits"], ht)
                n_val += 1

        val_loss         /= n_val
        shape_acc        /= n_val
        att_acc          /= n_val
        ht_acc           /= n_val
        nondom_shape_acc /= n_val

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val",   val_loss,   epoch)
        writer.add_scalar("acc/shape",  shape_acc,  epoch)
        writer.add_scalar("acc/att",    att_acc,    epoch)
        writer.add_scalar("acc/hand_type", ht_acc,  epoch)
        writer.add_scalar("lr", sched.get_last_lr()[0], epoch)
        if args.nondom_shape_head:
            writer.add_scalar("acc/nondom_shape", nondom_shape_acc, epoch)

        nd_str = f"  nd_shape={nondom_shape_acc:.3f}" if args.nondom_shape_head else ""
        print(f"Ep {epoch:3d}/{args.epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"shape={shape_acc:.3f}  att={att_acc:.3f}  ht={ht_acc:.3f}{nd_str}")

        if val_loss < best_val:
            best_val   = val_loss
            no_improve = 0
            torch.save({
                "epoch":             epoch,
                "val_loss":          val_loss,
                "model_state_dict":  model.state_dict(),
                "vocab_meta":        vocab_meta,
            }, out_dir / "best.pt")
            print(f"  ✓ saved best (val={val_loss:.4f})")
        elif args.patience > 0:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    writer.close()
    print(f"\nDone. Best val loss: {best_val:.4f}")
    print(f"Checkpoint: {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
