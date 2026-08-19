"""
Evaluate alignment CSVs against manual ground-truth annotations.

Metrics per boundary:
  MAE      — mean absolute error (frames)
  R@3/5/10 — fraction of boundaries within N frames of GT

Metrics per shape segment:
  mIoU   — mean intersection-over-union

Usage:
    python scripts/evaluate.py --annotations data/annotations2.json \\
        --test_list data/test_list2.json \\
        checkpoints/ctc/align_sts_v1.csv ...
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def load_alignment(csv_path: str) -> dict[str, list[tuple[str, int, int]]]:
    """
    Return {pose_fname: [(label, start, end), ...]} sorted by start, dominant hand.
    Deduplicates identical rows (from left-hand mirroring).
    """
    segs = defaultdict(set)
    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row.get('hand', 'dominant') != 'dominant':
                continue
            segs[row['pose_file']].add((
                row['label'], int(row['start_frame']), int(row['end_frame'])
            ))
    return {k: sorted(v, key=lambda x: x[1]) for k, v in segs.items()}


def extract_pred_boundaries(segs: list[tuple[str, int, int]], n_phases: int) -> list[int] | None:
    """
    Extract the n_phases+1 inner boundaries:
        [start_of_shape1, ..., start_of_shapeN, start_of_retract]

    Skips rest and <blank>. Handles missing __retract__ by using end of
    last shape segment as the final boundary.
    """
    inner = [(lbl, s, e) for lbl, s, e in segs
             if lbl not in ('rest', '<blank>') and lbl != '']
    if not inner:
        return None

    prep_idx = next((i for i, (l,_,_) in enumerate(inner) if l == '__prep__'), None)
    if prep_idx is None:
        return None

    retract_idx = next((i for i, (l,_,_) in enumerate(inner) if l == '__retract__'), None)

    if retract_idx is not None:
        signing = inner[prep_idx:retract_idx + 1]
        retract_start = signing[-1][1]
    else:
        signing = inner[prep_idx:]
        retract_start = None

    shape_segs = [(l, s, e) for l, s, e in signing
                  if l not in ('__prep__', '__retract__', '<blank>')]

    if len(shape_segs) != n_phases:
        return None

    if retract_start is not None:
        boundaries = [s for _, s, _ in shape_segs] + [retract_start]
    else:
        boundaries = [s for _, s, _ in shape_segs] + [shape_segs[-1][2]]

    return boundaries  # length n_phases+1


def iou(a_start, a_end, b_start, b_end) -> float:
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0.0


def collapse_repeated_shapes(shapes: list, boundaries: list):
    """Merge adjacent phases that share the same handshape label."""
    if not shapes:
        return shapes, boundaries
    out_shapes = [shapes[0]]
    out_bounds = [boundaries[0]]
    for i in range(1, len(shapes)):
        if shapes[i] == out_shapes[-1]:
            pass
        else:
            out_shapes.append(shapes[i])
            out_bounds.append(boundaries[i])
    out_bounds.append(boundaries[-1])
    return out_shapes, out_bounds


def evaluate(alignment: dict, annotations: dict, test_list: list):
    results_by_n = defaultdict(lambda: {'errors': [], 'ious': [],
                                         'n_clips': 0, 'n_missing': 0, 'n_mismatch': 0})
    test_fnames = {c['pose_fname'] for c in test_list}
    ann = {k: v for k, v in annotations.items() if k in test_fnames}

    for pose_fname, a in ann.items():
        shapes    = a.get('shapes', [])
        gt_bounds = a['boundaries']

        if shapes:
            shapes, gt_bounds = collapse_repeated_shapes(shapes, gt_bounds)
        n = len(shapes) if shapes else a['n_phases']
        res = results_by_n[n]
        res['n_clips'] += 1

        if pose_fname not in alignment:
            res['n_missing'] += 1
            continue

        pred_bounds = extract_pred_boundaries(alignment[pose_fname], n)
        if pred_bounds is None:
            res['n_mismatch'] += 1
            continue

        for gt, pr in zip(gt_bounds, pred_bounds):
            res['errors'].append(abs(gt - pr))

        for i in range(len(gt_bounds) - 1):
            res['ious'].append(iou(gt_bounds[i], gt_bounds[i+1],
                                   pred_bounds[i], pred_bounds[i+1]))

    all_e, all_i = [], []
    nc = nm = nmi = 0
    for res in results_by_n.values():
        all_e  += res['errors'];  all_i    += res['ious']
        nc     += res['n_clips']; nm       += res['n_missing']
        nmi    += res['n_mismatch']
    results_by_n['ALL'] = {'errors': all_e, 'ious': all_i,
                            'n_clips': nc, 'n_missing': nm, 'n_mismatch': nmi}
    return results_by_n


def fmt_row(res: dict) -> str:
    errs = res['errors']; ious = res['ious']
    n = res['n_clips']; miss = res['n_missing']; mism = res['n_mismatch']
    ev = n - miss - mism
    if not errs:
        return f"  n={ev:3d}/{n}  —"
    mae  = sum(errs) / len(errs)
    r3   = sum(1 for e in errs if e <=  3) / len(errs)
    r5   = sum(1 for e in errs if e <=  5) / len(errs)
    r10  = sum(1 for e in errs if e <= 10) / len(errs)
    miou = sum(ious) / len(ious) if ious else float('nan')
    return (f"  n={ev:3d}/{n}  MAE={mae:5.1f}f  "
            f"R@3={r3:.2f}  R@5={r5:.2f}  R@10={r10:.2f}  mIoU={miou:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csvs', nargs='+')
    ap.add_argument('--annotations', default='data/annotations2.json')
    ap.add_argument('--test_list',   default='data/test_list2.json')
    args = ap.parse_args()

    annotations = json.load(open(args.annotations))
    test_list   = json.load(open(args.test_list))
    test_fnames = {c['pose_fname'] for c in test_list}
    annotations = {k: v for k, v in annotations.items() if k in test_fnames}
    print(f"GT: {len(annotations)} test clips\n")

    for csv_path in args.csvs:
        name = Path(csv_path).stem
        alignment = load_alignment(csv_path)
        results   = evaluate(alignment, annotations, test_list)
        print(f"{'─'*65}")
        print(f"  {name}")
        print(f"{'─'*65}")
        for key in sorted(k for k in results if k != 'ALL') + ['ALL']:
            tag = f"{key}-phase" if key != 'ALL' else 'ALL    '
            print(f"  {tag}:{fmt_row(results[key])}")
        print()


if __name__ == '__main__':
    main()
