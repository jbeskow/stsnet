"""
Run STS-Net inference on a .pose or video (.mp4) file.

Outputs per-frame predictions for all 9 heads, or Viterbi alignment segments
when a description and sign window are provided.

Usage — per-frame predictions:
    python scripts/predict.py clip.pose --ckpt checkpoints/stsnet_base.pt
    python scripts/predict.py clip.mp4  --ckpt checkpoints/stsnet_base.pt

Usage — Viterbi alignment:
    python scripts/predict.py clip.pose \\
        --ckpt checkpoints/stsnet_base.pt \\
        --description "Flata handen, framåtriktad och nedåtvänd" \\
        --sign_start 12 --sign_end 58

Output format (per-frame, default):
    frame  state  shape  att  hand_type  contact_loc  contact_type  motion  nondom_shape  nondom_att
    0      rest   ...

Output format (alignment, --description):
    label           start  end
    rest            0      12
    __prep__        12     14
    Flata handen    14     53
    __retract__     53     58
    rest            58     72
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def fmt_table(rows: list[list], headers: list[str]) -> str:
    cols = [headers] + rows
    widths = [max(len(str(r[i])) for r in cols) for i in range(len(headers))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    lines = [fmt.format(*headers)]
    lines.append("  ".join("-" * w for w in widths))
    for row in rows:
        lines.append(fmt.format(*[str(v) for v in row]))
    return "\n".join(lines)


def extract_pose(video_path: Path, pose_path: Path) -> None:
    """Run video_to_pose (MediaPipe Holistic) on a video file."""
    try:
        from tqdm import tqdm
        bar = tqdm(total=1, desc="Extracting pose", unit="clip",
                   bar_format="{l_bar}{bar}| {elapsed}", leave=False)
    except ImportError:
        bar = None

    result = subprocess.run(
        ["video_to_pose", "-i", str(video_path), "-o", str(pose_path),
         "--format", "mediapipe"],
        capture_output=True,
    )
    if bar:
        bar.update(1)
        bar.close()

    if result.returncode != 0:
        msg = result.stderr.decode()[-400:]
        print(f"Error: pose extraction failed:\n{msg}", file=sys.stderr)
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(
        description="STS-Net inference on a .pose or video file")
    ap.add_argument("input_file", help="Input .pose file or video (.mp4, .mov, ...)")
    ap.add_argument("--ckpt", default="checkpoints/stsnet_base.pt",
                    help="Checkpoint path (default: checkpoints/stsnet_base.pt)")
    ap.add_argument("--device", default="cpu",
                    help="Torch device (default: cpu; use cuda for GPU)")
    ap.add_argument("--handedness", default="right", choices=["right", "left"])
    # Alignment mode
    ap.add_argument("--description", default=None,
                    help="Sign description for Viterbi alignment mode")
    ap.add_argument("--sign_start", type=int, default=None,
                    help="First frame of signing window (required for alignment)")
    ap.add_argument("--sign_end",   type=int, default=None,
                    help="Last frame (exclusive) of signing window (required for alignment)")
    # Per-frame output options
    ap.add_argument("--heads", nargs="+",
                    default=["state", "shape", "att", "hand_type",
                             "contact_loc", "contact_type", "motion",
                             "nondom_shape", "nondom_att"],
                    help="Heads to include in per-frame output")
    ap.add_argument("--start", type=int, default=None,
                    help="First frame to print in per-frame mode")
    ap.add_argument("--end",   type=int, default=None,
                    help="Last frame (exclusive) to print in per-frame mode")
    args = ap.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"Error: checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    # ── Pose extraction (video input) ─────────────────────────────────────────
    tmp_dir = None
    if input_path.suffix.lower() in VIDEO_SUFFIXES:
        tmp_dir = tempfile.TemporaryDirectory()
        pose_path = Path(tmp_dir.name) / (input_path.name + ".pose")
        extract_pose(input_path, pose_path)
    else:
        pose_path = input_path

    # ── Model loading ─────────────────────────────────────────────────────────
    try:
        from tqdm import tqdm
        bar = tqdm(total=1, desc="Loading model", unit="ckpt",
                   bar_format="{l_bar}{bar}| {elapsed}", leave=False)
    except ImportError:
        bar = None

    from stsnet.inference import STSNetInference
    model = STSNetInference(ckpt_path, device=args.device)

    if bar:
        bar.update(1)
        bar.close()

    # ── Alignment mode ─────────────────────────────────────────────────────────
    if args.description is not None:
        if args.sign_start is None or args.sign_end is None:
            print("Error: --sign_start and --sign_end are required for alignment mode",
                  file=sys.stderr)
            sys.exit(1)
        segs = model.align_clip(
            pose_path, args.description,
            args.sign_start, args.sign_end,
            handedness=args.handedness,
        )
        if tmp_dir:
            tmp_dir.cleanup()
        if not segs:
            print("Alignment failed (pose load or description parse error)", file=sys.stderr)
            sys.exit(1)
        rows = [[label, start, end] for label, start, end in segs]
        print(fmt_table(rows, ["label", "start", "end"]))
        return

    # ── Per-frame mode ─────────────────────────────────────────────────────────
    preds = model.predict_clip_decoded(pose_path, handedness=args.handedness)
    if tmp_dir:
        tmp_dir.cleanup()
    if not preds:
        print("Error: failed to load pose file", file=sys.stderr)
        sys.exit(1)

    heads = [h for h in args.heads if h in preds]
    T = len(next(iter(preds.values())))
    t0 = args.start or 0
    t1 = args.end   or T

    rows = [
        [t] + [preds[h][t] for h in heads]
        for t in range(t0, t1)
    ]
    print(fmt_table(rows, ["frame"] + heads))


if __name__ == "__main__":
    main()
