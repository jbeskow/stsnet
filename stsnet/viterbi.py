"""
Viterbi forced alignment utilities for STS-Net.

Blank-free Viterbi forced alignment (ctc_forced_align) plus helper
functions and heuristic boundary constants.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PREP_LABEL    = "__prep__"
RETRACT_LABEL = "__retract__"

# Calibrated from annotations.json + annotations2.json (analyze_retraction.py)
# Prep: fixed fraction of window (low variance, works well)
# Retract: fixed duration in frames (beats displacement plateau; median=9f, std=3.5f)
HEURISTIC_PARAMS = {
    "prep_frac":        0.111,
    "retract_frac":     0.172,   # fallback fraction when pose cache unavailable
    "retract_dur":      9,       # fixed retract duration (frames)
    "prep_plateau":     0.762,
    "retract_plateau":  0.719,
    "prep_vel_frac":    0.592,
    "retract_vel_frac": 0.395,
}


# ---------------------------------------------------------------------------
# Viterbi forced CTC alignment
# ---------------------------------------------------------------------------

def ctc_forced_align(
    log_probs: np.ndarray,  # (T, C) float32 — output of log_softmax
    targets:   list[int],   # (L,) — token indices (no repeated consecutive tokens)
    blank:     int,         # unused — kept for API compatibility
    min_dur:   int = 1,     # minimum frames per token
) -> np.ndarray:
    """
    Blank-free Viterbi forced alignment.

    Every frame is assigned to exactly one target token — no blank states.
    This avoids the blank-absorption problem that makes standard CTC Viterbi
    produce peaky (1-2 frame) alignments when blank posteriors dominate.

    Since sign language targets never have repeated consecutive tokens
    (PREP != shape != RETRACT), blanks are unnecessary for the alignment step.
    The model may still have been trained with CTC blank; we simply ignore the
    blank column of log_probs during alignment.

    min_dur: token s can only advance to s+1 after min_dur consecutive frames.

    Returns frame_labels (T,) int16: values in [0, L-1] (index into targets).
    All frames are assigned; no -1 (blank) values are returned.
    """
    T = log_probs.shape[0]
    L = len(targets)

    NEG_INF = np.float32(-1e30)

    # dp[t, s]   = best log-prob for path ending at token s at time t
    # back[t, s] = predecessor state
    # dur[t, s]  = consecutive frames in state s at time t
    dp   = np.full((T, L), NEG_INF, dtype=np.float32)
    back = np.full((T, L), -1,      dtype=np.int16)
    dur  = np.zeros((T, L),         dtype=np.int16)

    # Initialise at t=0: can start in state 0 only
    dp[0, 0]   = log_probs[0, targets[0]]
    dur[0, 0]  = 1

    for t in range(1, T):
        lp = log_probs[t]
        for s in range(L):
            # Stay in current state (always allowed)
            best_p = dp[t-1, s]
            best_s = s

            # Advance from s-1 (only if s-1 has been held for >= min_dur frames)
            if s >= 1 and dur[t-1, s-1] >= min_dur:
                if dp[t-1, s-1] > best_p:
                    best_p = dp[t-1, s-1]
                    best_s = s - 1

            if best_p > NEG_INF:
                dp[t, s]   = best_p + lp[targets[s]]
                back[t, s] = best_s
                dur[t, s]  = dur[t-1, best_s] + 1 if best_s == s else 1

    # Backtrack — must end at state L-1
    s = L - 1
    path = np.empty(T, dtype=np.int16)
    path[T-1] = s
    for t in range(T-1, 0, -1):
        s = back[t, s]
        path[t-1] = s

    return path  # values in [0, L-1], index into targets list


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def frame_labels_to_segments(
    frame_labels: np.ndarray,   # (T,): target indices or -1 (blank)
    target_names: list[str],    # length L: name for each target index
    sign_start:   int,
    full_len:     int,
) -> list[tuple[str, int, int]]:
    """
    Convert per-frame target indices to (label, start_frame, end_frame) segments.
    frame_labels indices are relative to the signing window; sign_start is added
    to convert to absolute clip frame indices.  Rest segments are prepended /
    appended for frames outside the signing window.
    """
    T    = len(frame_labels)
    segs = []

    # Leading rest
    if sign_start > 0:
        segs.append(("rest", 0, sign_start))

    if T > 0:
        prev_idx   = int(frame_labels[0])
        prev_start = 0
        for i in range(1, T):
            cur = int(frame_labels[i])
            if cur != prev_idx:
                label = target_names[prev_idx] if prev_idx >= 0 else "<blank>"
                segs.append((label, prev_start + sign_start, i + sign_start))
                prev_idx   = cur
                prev_start = i
        label = target_names[prev_idx] if prev_idx >= 0 else "<blank>"
        segs.append((label, prev_start + sign_start, T + sign_start))

    # Trailing rest
    end = sign_start + T
    if end < full_len:
        segs.append(("rest", end, full_len))

    return segs


def frame_labels_to_inner_segs(
    frame_labels: np.ndarray,   # (T,) target indices in [0, L-1]
    target_names: list[str],    # length L
    offset: int,                # prep_end (absolute frame)
) -> list[tuple[str, int, int]]:
    """Convert per-frame labels to segments; offset converts to absolute frames."""
    T = len(frame_labels)
    if T == 0:
        return []
    segs = []
    prev_idx   = int(frame_labels[0])
    prev_start = 0
    for i in range(1, T):
        cur = int(frame_labels[i])
        if cur != prev_idx:
            segs.append((target_names[prev_idx], prev_start + offset, i + offset))
            prev_idx   = cur
            prev_start = i
    segs.append((target_names[prev_idx], prev_start + offset, T + offset))
    return segs


def equal_spacing_inner(
    shapes:        list[str],
    prep_end:      int,
    retract_start: int,
) -> list[tuple[str, int, int]]:
    """Equal-length shape segments within [prep_end, retract_start]."""
    win = retract_start - prep_end
    n   = len(shapes)
    if n == 0:
        return []
    seg_w = max(1, win // n)
    segs  = []
    cur   = prep_end
    for k, sh in enumerate(shapes):
        nxt = cur + seg_w if k < n - 1 else retract_start
        nxt = min(nxt, retract_start)
        if nxt > cur:
            segs.append((sh, cur, nxt))
        cur = nxt
    return segs


def equal_spacing_fallback(
    shapes:     list[str],
    sign_start: int,
    sign_end:   int,
    full_len:   int,
) -> list[tuple[str, int, int]]:
    """Fallback when window is too short for CTC: equal-length segments."""
    segs = []
    if sign_start > 0:
        segs.append(("rest", 0, sign_start))

    win    = sign_end - sign_start
    labels = [PREP_LABEL] + shapes + [RETRACT_LABEL]
    n      = len(labels)
    seg_w  = max(1, win // n)
    cur    = sign_start
    for k, lbl in enumerate(labels):
        nxt = cur + seg_w if k < n - 1 else sign_end
        nxt = min(nxt, sign_end)
        if nxt > cur:
            segs.append((lbl, cur, nxt))
        cur = nxt

    if sign_end < full_len:
        segs.append(("rest", sign_end, full_len))
    return segs
