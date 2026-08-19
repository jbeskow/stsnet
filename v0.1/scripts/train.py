"""
Train STSNet: per-frame multi-head phonology model on SSLLMultiHeadDataset.

All heads use cross-entropy with ignore_index=-1.
Loss = state + shape + att + hand_type + contact_loc + contact_type + motion
       [+ nondom_shape + nondom_att]   (weighted)

Usage:
    conda run -n slp --no-capture-output python -u scripts/train.py \\
        --alignment checkpoints/ctc/align_seed.csv \\
        --ckpt_dir  checkpoints/stsnet_v1 \\
        --streams dom nondom body face \\
        --bilstm_layers 1
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from stsnet.data.pose_io import set_pose_cache_dir
from stsnet.data.description import load_llm_parse_cache
from stsnet.data.multihead import SSLLMultiHeadDataset
from stsnet.model import STSNet
from stsnet.train_utils import (
    HEADS,
    collate_pad,
    signer_split,
    compute_losses,
    compute_accs,
    evaluate,
)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    set_pose_cache_dir(args.cache_dir)
    if args.llm_cache:
        load_llm_parse_cache(args.llm_cache)

    ds = SSLLMultiHeadDataset(
        csv_path      = args.csv_path,
        pose_dir      = args.pose_dir,
        vocab_file    = args.vocab_file,
        alignment_csv = args.alignment,
        mirror_left   = True,
        noise_std     = args.noise_std,
        llm_cache     = args.llm_cache,
        signer_map    = args.signer_map,
    )

    train_idx, val_idx = signer_split(ds, val_frac=args.val_frac)
    train_ds = Subset(ds, train_idx)
    val_ds   = Subset(ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_pad,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_pad,
                              pin_memory=True)

    model_kwargs = dict(
        num_shapes        = ds.num_shapes,
        num_atts          = ds.num_atts,
        num_contact_locs  = ds.num_cloc,
        num_contact_types = ds.num_ctype,
        num_nondom_shapes = ds.num_shapes,
        num_nondom_atts   = ds.num_atts,
        hidden_dim        = args.hidden_dim,
        conv_layers       = args.conv_layers,
        kernel_size       = args.kernel_size,
        dropout           = args.dropout,
        bilstm_layers     = args.bilstm_layers,
        streams           = tuple(args.streams),
    )

    model = STSNet(**model_kwargs)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}  streams={args.streams}  bilstm={args.bilstm_layers}")

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr / 20)

    weights = {
        "state": args.w_state, "shape": 1.0, "att": args.w_att,
        "hand_type": args.w_hand_type,
        "contact_loc": args.w_contact, "contact_type": args.w_contact,
        "motion": args.w_motion,
        "nondom_shape": args.w_nondom, "nondom_att": args.w_nondom,
    }

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(ckpt_dir / "tb"))

    best_val_shape_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        sum_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            out  = model(batch["dominant"], batch["nondominant"], batch["body"],
                         face=batch.get("face"), lengths=batch.get("lengths"))
            ls   = compute_losses(out, batch, weights)
            loss = ls["total"]
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            sum_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = sum_loss / max(n_batches, 1)

        val_losses, val_accs = evaluate(model, val_loader, device, weights)
        val_shape_acc = val_accs.get("shape", 0.0)

        writer.add_scalar("train/loss",       avg_train_loss, epoch)
        writer.add_scalar("val/loss",         val_losses["total"], epoch)
        for k, acc in val_accs.items():
            writer.add_scalar(f"val/acc_{k}", acc, epoch)

        acc_str = "  ".join(
            f"{k}={val_accs[k]:.3f}"
            for k in ["state", "shape", "att", "hand_type",
                      "contact_loc", "contact_type", "motion",
                      "nondom_shape", "nondom_att"]
            if k in val_accs
        )
        print(f"Ep {epoch:3d}/{args.epochs}  train={avg_train_loss:.4f}"
              f"  val={val_losses['total']:.4f}  {acc_str}")

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_shape_acc": val_shape_acc,
            "args": vars(args),
            "vocabs": {
                "shape_to_idx":  ds.shape_to_idx,
                "att_to_idx":    ds.att_to_idx,
                "state_to_idx":  ds.state_to_idx,
                "motion_to_idx": ds.motion_to_idx,
                "cloc_to_idx":   ds.cloc_to_idx,
                "ctype_to_idx":  ds.ctype_to_idx,
            },
        }
        torch.save(ckpt, ckpt_dir / "last.pt")
        if val_shape_acc > best_val_shape_acc:
            best_val_shape_acc = val_shape_acc
            torch.save(ckpt, ckpt_dir / "best.pt")
            print(f"  → best shape acc: {best_val_shape_acc:.4f}")

    writer.close()
    print(f"Done. Best shape acc: {best_val_shape_acc:.4f}")


def main():
    ap = argparse.ArgumentParser()
    # Data
    ap.add_argument("--csv_path",    default="/nfs/signbot1/data/SSLL/sign_data_with_signer_fr.csv")
    ap.add_argument("--pose_dir",    default="/nfs/signbot1/data/SSLL/pose")
    ap.add_argument("--cache_dir",   default="/nfs/signbot1/data/SSLL/pose_cache")
    ap.add_argument("--vocab_file",  default="/nfs/signbot1/data/SSLL/sts_handformer.txt")
    ap.add_argument("--alignment",   required=True)
    ap.add_argument("--signer_map",  default=None,
                    help="CSV mapping video_id->signer (overrides signer column in csv_path)")
    ap.add_argument("--llm_cache",   default=None)
    # Model
    ap.add_argument("--streams",     nargs="+",
                    default=["dom", "nondom", "body", "face"],
                    choices=["dom", "nondom", "body", "face"])
    ap.add_argument("--hidden_dim",  type=int,   default=512)
    ap.add_argument("--conv_layers", type=int,   default=3)
    ap.add_argument("--kernel_size", type=int,   default=5)
    ap.add_argument("--dropout",     type=float, default=0.2)
    ap.add_argument("--bilstm_layers",type=int,  default=1)
    # Training
    ap.add_argument("--ckpt_dir",    required=True)
    ap.add_argument("--epochs",      type=int,   default=60)
    ap.add_argument("--batch_size",  type=int,   default=32)
    ap.add_argument("--lr",          type=float, default=3e-4)
    ap.add_argument("--weight_decay",type=float, default=1e-4)
    ap.add_argument("--grad_clip",   type=float, default=1.0)
    ap.add_argument("--val_frac",    type=float, default=0.15)
    ap.add_argument("--num_workers", type=int,   default=0)
    ap.add_argument("--noise_std",   type=float, default=0.005)
    # Loss weights (shape always = 1.0)
    ap.add_argument("--w_state",     type=float, default=1.0)
    ap.add_argument("--w_att",       type=float, default=1.0)
    ap.add_argument("--w_hand_type", type=float, default=0.5)
    ap.add_argument("--w_contact",   type=float, default=0.5)
    ap.add_argument("--w_motion",    type=float, default=0.5)
    ap.add_argument("--w_nondom",    type=float, default=0.5)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
