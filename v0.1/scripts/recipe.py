"""
STS-Net training recipe — cold start, no pre-trained models or existing alignments.

Recipe:
  Round 0 : generate seed alignment from pseudo_signing.json
            (heuristic prep/retract + equal-split shapes, no model)
  Round 1+ : train STSNet [all streams, no BiLSTM] → multi-head Viterbi align
             repeat until mIoU converges (Δ < CONV_THRESHOLD)
  Final    : train STSNet [all streams, BiLSTM] on converged alignment

Progress is written to logs/recipe_progress.log in real time.
Checkpoints: checkpoints/stsnet_recipe/rN_nobilstm/  and  .../final_bilstm/
Alignments:  checkpoints/ctc/align_recipe_rN.csv

GPU: set via CUDA_VISIBLE_DEVICES or --gpu argument.
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT      = Path(__file__).parent.parent
CONDA_ENV = "slp"
LOG       = ROOT / "logs/recipe_progress.log"

SEED_ALIGN     = ROOT / "checkpoints/ctc/align_seed.csv"
ANNOT          = ROOT / "data/annotations2.json"
TEST_LIST      = ROOT / "data/test_list2.json"
CONV_THRESHOLD = 0.005
MAX_ROUNDS     = 6

COMMON = dict(
    csv_path    = "/nfs/signbot1/data/SSLL/sign_data_with_signer_fr.csv",
    signer_map  = ROOT / "data/signer_map.csv",
    pose_dir    = "/nfs/signbot1/data/SSLL/pose",
    cache_dir   = "/nfs/signbot1/data/SSLL/pose_cache",
    vocab_file  = ROOT / "data/sts_handformer.txt",
    hidden_dim  = 512,
    conv_layers = 3,
    kernel_size = 5,
    epochs      = 60,
    batch_size  = 32,
    lr          = "3e-4",
    weight_decay= "1e-4",
    grad_clip   = 1.0,
    val_frac    = 0.15,
    num_workers = 0,
    noise_std   = 0.005,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg: str):
    ts  = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def log_section(title: str):
    bar = "=" * 60
    log(bar)
    log(title)
    log(bar)


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def run_bg(cmd: str, logfile: Path, gpu: int, append=False) -> subprocess.Popen:
    mode = "a" if append else "w"
    f    = open(logfile, mode)
    env  = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    return subprocess.Popen(
        f"conda run -n {CONDA_ENV} --no-capture-output {cmd}",
        shell=True, stdout=f, stderr=f, env=env, cwd=ROOT,
    )


def wait(proc: subprocess.Popen, logfile: Path, label: str,
         timeout_min=180, poll_sec=60) -> bool:
    start = time.time()
    last_ep = ""
    while proc.poll() is None:
        if (time.time() - start) / 60 > timeout_min:
            log(f"  [{label}] TIMEOUT — killing")
            proc.kill(); return False
        try:
            lines = logfile.read_text(errors="replace").splitlines()
            ep = [l for l in lines if l.startswith("Ep ")]
            if ep and ep[-1] != last_ep:
                last_ep = ep[-1]
                log(f"  [{label}] {last_ep}")
        except Exception:
            pass
        time.sleep(poll_sec)
    ok = proc.returncode == 0
    if not ok:
        log(f"  [{label}] FAILED (rc={proc.returncode})")
    return ok


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(name: str, align_csv: Path, bilstm: bool, gpu: int) -> Path | None:
    ckpt_dir = ROOT / "checkpoints/stsnet_recipe" / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logfile  = ROOT / "logs" / f"recipe_{name}.log"

    bilstm_layers = 1 if bilstm else 0
    streams = "dom nondom body face"

    args_str = (
        f"python -u scripts/train.py"
        f" --alignment {align_csv}"
        f" --ckpt_dir {ckpt_dir}"
        f" --streams {streams}"
        f" --bilstm_layers {bilstm_layers}"
        f" --hidden_dim {COMMON['hidden_dim']}"
        f" --conv_layers {COMMON['conv_layers']}"
        f" --kernel_size {COMMON['kernel_size']}"
        f" --epochs {COMMON['epochs']}"
        f" --batch_size {COMMON['batch_size']}"
        f" --lr {COMMON['lr']}"
        f" --weight_decay {COMMON['weight_decay']}"
        f" --grad_clip {COMMON['grad_clip']}"
        f" --val_frac {COMMON['val_frac']}"
        f" --num_workers {COMMON['num_workers']}"
        f" --noise_std {COMMON['noise_std']}"
        f" --csv_path {COMMON['csv_path']}"
        f" --signer_map {COMMON['signer_map']}"
        f" --pose_dir {COMMON['pose_dir']}"
        f" --cache_dir {COMMON['cache_dir']}"
        f" --vocab_file {COMMON['vocab_file']}"
    )

    log(f"  Train [{name}] bilstm={bilstm}  GPU={gpu}")
    proc = run_bg(args_str, logfile, gpu)
    ok   = wait(proc, logfile, name)

    best = ckpt_dir / "best.pt"
    if ok and best.exists():
        try:
            lines = logfile.read_text(errors="replace").splitlines()
            ep_lines = [l for l in lines if l.startswith("Ep ")]
            if ep_lines:
                log(f"  [{name}] final: {ep_lines[-1]}")
        except Exception:
            pass
        return best
    return None


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def align(ckpt: Path, name: str, gpu: int) -> Path | None:
    out_csv = ROOT / "checkpoints/ctc" / f"align_recipe_{name}.csv"
    logfile = ROOT / "logs" / f"recipe_align_{name}.log"

    cmd = (
        f"python -u scripts/align.py"
        f" --ckpt {ckpt}"
        f" --pseudo_signing pseudo_signing.json"
        f" --output {out_csv}"
        f" --cache_dir {COMMON['cache_dir']}"
        f" --csv_path {COMMON['csv_path']}"
        f" --pose_dir {COMMON['pose_dir']}"
        f" --vocab_file {COMMON['vocab_file']}"
        f" --max_shapes 3 --min_dur 3 --batch_size 64"
    )
    log(f"  Align [{name}] GPU={gpu}")
    proc = run_bg(cmd, logfile, gpu)
    proc.wait()
    if out_csv.exists():
        log(f"  [{name}] alignment done → {out_csv.name}")
        return out_csv
    log(f"  [{name}] alignment FAILED")
    return None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_alignment(align_csv: Path) -> dict:
    r = subprocess.run(
        f"conda run -n {CONDA_ENV} --no-capture-output"
        f" python scripts/evaluate.py {align_csv}"
        f" --annotations {ANNOT} --test_list {TEST_LIST}",
        shell=True, cwd=ROOT, capture_output=True, text=True,
    )
    out = r.stdout + r.stderr
    m = {}
    for ph in [2, 3]:
        mo = re.search(rf"{ph}-phase:.*n=\s*(\d+)/\d+.*MAE=\s*([\d.]+)f.*mIoU=([\d.]+)", out)
        if mo:
            m[f"ph{ph}_n"]    = int(mo.group(1))
            m[f"ph{ph}_mae"]  = float(mo.group(2))
            m[f"ph{ph}_miou"] = float(mo.group(3))
    mo = re.search(r"ALL\s*:.*n=\s*(\d+)/\d+.*MAE=\s*([\d.]+)f.*mIoU=([\d.]+)", out)
    if mo:
        m["all_n"]    = int(mo.group(1))
        m["all_mae"]  = float(mo.group(2))
        m["all_miou"] = float(mo.group(3))
    return m


def log_eval(label: str, m: dict):
    log(f"  {label}: "
        f"2ph mIoU={m.get('ph2_miou','?'):.3f}  "
        f"3ph mIoU={m.get('ph3_miou','?'):.3f}  "
        f"ALL mIoU={m.get('all_miou','?'):.3f}  "
        f"MAE={m.get('all_mae','?'):.1f}f  "
        f"n={m.get('all_n','?')}/100")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=4)
    args = ap.parse_args()
    GPU = args.gpu

    LOG.parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "checkpoints/stsnet_recipe").mkdir(parents=True, exist_ok=True)
    (ROOT / "checkpoints/ctc").mkdir(parents=True, exist_ok=True)

    log_section("STS-Net recipe — START")
    log(f"GPU={GPU}  max_rounds={MAX_ROUNDS}  conv_threshold={CONV_THRESHOLD}")
    log(f"Seed alignment: {SEED_ALIGN.name}")

    # Round 0: generate cold-start seed alignment
    log_section("Round 0: generate seed alignment (pseudo_signing + equal split)")
    seed_cmd = (
        f"python -u scripts/make_seed_alignment.py"
        f" --pseudo_signing pseudo_signing.json"
        f" --output {SEED_ALIGN}"
        f" --csv_path {COMMON['csv_path']}"
        f" --vocab_file {COMMON['vocab_file']}"
        f" --max_shapes 3"
    )
    r = subprocess.run(
        f"conda run -n {CONDA_ENV} --no-capture-output {seed_cmd}",
        shell=True, cwd=ROOT, capture_output=True, text=True,
    )
    log((r.stdout + r.stderr).strip().splitlines()[-1] if (r.stdout + r.stderr).strip() else "seed done")
    if not SEED_ALIGN.exists():
        log("Seed alignment generation failed — aborting"); return

    m0 = evaluate_alignment(SEED_ALIGN)
    log_eval("seed (equal-split heuristic)", m0)
    prev_miou  = m0.get("all_miou", 0.0)
    cur_align  = SEED_ALIGN

    # Bootstrap rounds: noBiLSTM
    final_align = cur_align
    for rnd in range(1, MAX_ROUNDS + 1):
        log_section(f"Round {rnd}: train noBiLSTM → align")

        ckpt = train(f"r{rnd}_nobilstm", cur_align, bilstm=False, gpu=GPU)
        if ckpt is None:
            log(f"  Round {rnd}: training failed — stopping bootstrap"); break

        new_align = align(ckpt, f"r{rnd}", GPU)
        if new_align is None:
            log(f"  Round {rnd}: alignment failed — stopping bootstrap"); break

        m = evaluate_alignment(new_align)
        log_eval(f"r{rnd}", m)
        final_align = new_align

        delta = abs(m.get("all_miou", 0.0) - prev_miou)
        log(f"  Δ mIoU = {delta:.4f}")
        if delta < CONV_THRESHOLD:
            log("  Converged."); break
        prev_miou = m.get("all_miou", prev_miou)
        cur_align = new_align

    # Final: BiLSTM
    log_section("Final: train BiLSTM on converged alignment")
    final_ckpt = train("final_bilstm", final_align, bilstm=True, gpu=GPU)

    if final_ckpt:
        log_section("Final evaluation")
        final_bilstm_align = align(final_ckpt, "final_bilstm", GPU)
        if final_bilstm_align:
            m = evaluate_alignment(final_bilstm_align)
            log_eval("final BiLSTM alignment", m)
        log(f"Best checkpoint: {final_ckpt}")

    log_section("DONE")
    log(f"Progress log: {LOG}")


if __name__ == "__main__":
    main()
