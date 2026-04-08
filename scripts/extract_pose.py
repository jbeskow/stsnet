"""
Extract MediaPipe Holistic poses from SSLL mp4 files.

Reads the SSLL CSV to enumerate which videos to process, then runs
`video_to_pose` (pose_format / MediaPipe Holistic) on each.

Output: <pose_dir>/<basename>.mp4.pose
  e.g.  pose/mossa-00003-tecken.mp4.pose

This is a prerequisite step, not part of the training recipe.

Usage:
    conda run -n slp python scripts/extract_pose.py \\
        --csv_path  /path/to/sign_data.csv \\
        --video_dir /path/to/SSLL           \\
        --pose_dir  /path/to/SSLL/pose      \\
        --workers   4
"""

import argparse
import csv
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def _extract(args: tuple[str, str]) -> tuple[str, bool, str]:
    video_path_str, pose_path_str = args
    result = subprocess.run(
        ["video_to_pose", "-i", video_path_str, "-o", pose_path_str,
         "--format", "mediapipe"],
        capture_output=True,
    )
    name = Path(video_path_str).name
    if result.returncode == 0:
        return name, True, ""
    return name, False, result.stderr.decode()[-400:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path",  required=True,
                    help="SSLL metadata CSV (sign_data_with_signer_fr.csv or equivalent)")
    ap.add_argument("--video_dir", required=True,
                    help="Root directory containing SSLL mp4 files "
                         "(movie paths in CSV are relative to this)")
    ap.add_argument("--pose_dir",  required=True,
                    help="Output directory for .pose files")
    ap.add_argument("--workers",   type=int, default=4)
    args = ap.parse_args()

    video_root = Path(args.video_dir)
    pose_dir   = Path(args.pose_dir)
    pose_dir.mkdir(parents=True, exist_ok=True)

    # Collect unique video paths from CSV
    videos = {}  # basename -> full_path
    with open(args.csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            movie = row.get("movie", "").strip()
            if not movie or movie == "nan":
                continue
            full = video_root / movie
            stem = Path(movie).name          # e.g. mossa-00003-tecken.mp4
            if stem not in videos:
                videos[stem] = full

    todo = [
        (str(path), str(pose_dir / (stem + ".pose")))
        for stem, path in sorted(videos.items())
        if not (pose_dir / (stem + ".pose")).exists()
    ]
    skip = len(videos) - len(todo)
    print(f"Videos in CSV: {len(videos)}  |  already done: {skip}  |  to extract: {len(todo)}")
    if not todo:
        print("Nothing to do.")
        return

    print(f"Running {args.workers} parallel workers")
    ok = fail = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_extract, t): t for t in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            name, success, err = fut.result()
            if success:
                ok += 1
            else:
                fail += 1
                print(f"\nFAIL {name}: {err}", file=sys.stderr)
            if i % 10 == 0 or i == len(todo):
                print(f"  [{i}/{len(todo)}] {ok} ok, {fail} failed", flush=True)

    print(f"\nDone. {ok} extracted, {skip} already existed, {fail} failed.")


if __name__ == "__main__":
    main()
