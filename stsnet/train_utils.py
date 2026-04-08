"""
Training utilities for STS-Net: losses, accuracies, evaluation, collation, and splitting.
"""

import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Head definitions
# ---------------------------------------------------------------------------

HEADS = [
    # (name,         logit_key,               label_key,      ignore)
    ("state",        "state_logits",          "state",        -1),
    ("shape",        "shape_logits",          "shape",        -1),
    ("att",          "att_logits",            "att",          -1),
    ("hand_type",    "hand_type_logits",      "hand_type",    -1),
    ("contact_loc",  "contact_loc_logits",    "contact_loc",  -1),
    ("contact_type", "contact_type_logits",   "contact_type", -1),
    ("motion",       "motion_logits",         "motion",       -1),
    ("nondom_shape", "nondom_shape_logits",   "nondom_shape", -1),
    ("nondom_att",   "nondom_att_logits",     "nondom_att",   -1),
]


# ---------------------------------------------------------------------------
# Collate (pad to longest sequence in batch)
# ---------------------------------------------------------------------------

def collate_pad(batch: list[dict]) -> dict:
    str_keys = {"signer", "word"}
    tensor_keys = [k for k in batch[0] if k not in str_keys]
    out = {k: [b[k] for b in batch] for k in str_keys}
    for k in tensor_keys:
        tensors = [b[k] for b in batch]
        if tensors[0].dim() == 0:
            out[k] = torch.stack(tensors)
        else:
            T_max = max(t.shape[0] for t in tensors)
            pad_val = -1 if tensors[0].dtype == torch.long else 0.0
            padded = torch.full((len(tensors), T_max, *tensors[0].shape[1:]),
                                pad_val, dtype=tensors[0].dtype)
            for i, t in enumerate(tensors):
                padded[i, :t.shape[0]] = t
            out[k] = padded
    out["lengths"] = torch.tensor([b["dominant"].shape[0] for b in batch])
    return out


# ---------------------------------------------------------------------------
# Signer-based train/val split
# ---------------------------------------------------------------------------

def signer_split(dataset, val_frac=0.15, seed=42):
    """Split dataset into train/val index lists by signer."""
    signers = sorted({s["signer"] for s in dataset.samples})
    rng = random.Random(seed)
    rng.shuffle(signers)
    n_val = max(1, round(len(signers) * val_frac))
    val_signers = set(signers[:n_val])
    train_idx, val_idx = [], []
    for i, s in enumerate(dataset.samples):
        (val_idx if s["signer"] in val_signers else train_idx).append(i)
    return train_idx, val_idx


# ---------------------------------------------------------------------------
# Loss / accuracy helpers
# ---------------------------------------------------------------------------

def frame_ce(logits, targets, valid, ignore=-1):
    mask = valid & (targets != ignore)
    if mask.sum() == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(logits[mask], targets[mask])


def frame_acc(logits, targets, valid, ignore=-1):
    mask = valid & (targets != ignore)
    if mask.sum() == 0:
        return 0, 0
    preds = logits[mask].argmax(-1)
    return int((preds == targets[mask]).sum()), int(mask.sum())


def compute_losses(outputs, batch, weights):
    valid = batch["valid"]   # (B, T) bool from dataset
    losses = {}
    total = None
    for name, logit_key, label_key, ignore in HEADS:
        if logit_key not in outputs:
            continue
        loss = frame_ce(outputs[logit_key], batch[label_key], valid, ignore)
        losses[name] = loss
        w = weights.get(name, 1.0)
        total = loss * w if total is None else total + loss * w
    losses["total"] = total if total is not None else torch.tensor(0.0)
    return losses


def compute_accs(outputs, batch):
    valid = batch["valid"]
    accs = {}
    for name, logit_key, label_key, ignore in HEADS:
        if logit_key not in outputs:
            accs[name] = (0, 0)
            continue
        accs[name] = frame_acc(outputs[logit_key], batch[label_key], valid, ignore)
    return accs


@torch.no_grad()
def evaluate(model, loader, device, weights):
    model.eval()
    sum_losses = {}
    sum_accs   = {n: (0, 0) for n, *_ in HEADS}
    n = 0
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        out = model(batch["dominant"], batch["nondominant"], batch["body"],
                    face=batch.get("face"), lengths=batch.get("lengths"))
        ls  = compute_losses(out, batch, weights)
        acs = compute_accs(out, batch)
        for k, v in ls.items():
            sum_losses[k] = sum_losses.get(k, 0.0) + v.item()
        for k, (c, t) in acs.items():
            c0, t0 = sum_accs[k]
            sum_accs[k] = (c0 + c, t0 + t)
        n += 1
    avg_losses = {k: v / max(n, 1) for k, v in sum_losses.items()}
    avg_accs   = {k: c / max(t, 1) for k, (c, t) in sum_accs.items()}
    return avg_losses, avg_accs
