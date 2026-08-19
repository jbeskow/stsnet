"""
Unit tests for ctc_forced_align from stsnet.viterbi.
"""

import numpy as np
import unittest

from stsnet.viterbi import ctc_forced_align


def _uniform_log_probs(T: int, C: int) -> np.ndarray:
    """Return uniform log-probs (T, C)."""
    return np.full((T, C), -np.log(C), dtype=np.float32)


def _peaked_log_probs(T: int, targets: list[int], C: int, min_dur: int = 1) -> np.ndarray:
    """
    Return log-probs where each target token dominates in its expected segment.
    Segments are evenly distributed across frames.
    """
    lp = np.full((T, C), -100.0, dtype=np.float32)
    L = len(targets)
    seg_len = max(min_dur, T // L)
    for l, tok in enumerate(targets):
        s = l * seg_len
        e = (l + 1) * seg_len if l < L - 1 else T
        lp[s:e, tok] = 0.0   # log P = 0 → P = 1
    return lp


class TestMonotone(unittest.TestCase):
    """Output must be monotonically non-decreasing."""

    def test_monotone_random(self):
        rng = np.random.default_rng(0)
        T, C, L = 50, 10, 4
        lp = np.log(rng.dirichlet(np.ones(C), size=T).astype(np.float32))
        targets = list(range(L))
        path = ctc_forced_align(lp, targets, blank=C, min_dur=1)
        self.assertEqual(len(path), T)
        for t in range(1, T):
            self.assertGreaterEqual(
                int(path[t]), int(path[t - 1]),
                f"Non-monotone at t={t}: {path[t-1]} → {path[t]}"
            )

    def test_monotone_uniform(self):
        T, C = 30, 5
        targets = [0, 2, 4]
        lp = _uniform_log_probs(T, C)
        path = ctc_forced_align(lp, targets, blank=C)
        for t in range(1, T):
            self.assertGreaterEqual(int(path[t]), int(path[t - 1]))

    def test_starts_at_zero_ends_at_L_minus_1(self):
        T, C, L = 20, 6, 3
        targets = [0, 3, 5]
        lp = _uniform_log_probs(T, C)
        path = ctc_forced_align(lp, targets, blank=C)
        self.assertEqual(int(path[0]), 0, "Must start in state 0")
        self.assertEqual(int(path[-1]), L - 1, "Must end in final state")


class TestMinDur(unittest.TestCase):
    """No segment should be shorter than min_dur frames."""

    def test_min_dur_3(self):
        T, C = 60, 5
        targets = [0, 2, 4]
        lp = _uniform_log_probs(T, C)
        min_dur = 3
        path = ctc_forced_align(lp, targets, blank=C, min_dur=min_dur)
        # Count consecutive runs for each state
        runs = {}
        cur, count = int(path[0]), 1
        for t in range(1, T):
            if int(path[t]) == cur:
                count += 1
            else:
                runs.setdefault(cur, []).append(count)
                cur, count = int(path[t]), 1
        runs.setdefault(cur, []).append(count)
        for state, lengths in runs.items():
            for length in lengths:
                self.assertGreaterEqual(
                    length, min_dur,
                    f"State {state} run of length {length} < min_dur={min_dur}"
                )

    def test_min_dur_5(self):
        T, C = 100, 4
        targets = [0, 1, 2, 3]
        min_dur = 5
        lp = _uniform_log_probs(T, C)
        path = ctc_forced_align(lp, targets, blank=C, min_dur=min_dur)
        # Each state should appear at least min_dur times total
        for s in range(len(targets)):
            self.assertGreaterEqual(int(np.sum(path == s)), min_dur)


class TestSingleSegment(unittest.TestCase):
    """With a single target, all frames should be assigned that label."""

    def test_single_segment(self):
        T, C = 25, 3
        targets = [1]
        lp = _uniform_log_probs(T, C)
        path = ctc_forced_align(lp, targets, blank=C)
        self.assertEqual(path.shape, (T,))
        self.assertTrue(np.all(path == 0), "All frames should be assigned state index 0")

    def test_single_segment_peaked(self):
        T, C = 15, 4
        targets = [2]
        lp = _peaked_log_probs(T, targets, C)
        path = ctc_forced_align(lp, targets, blank=C)
        self.assertTrue(np.all(path == 0))


class TestKnownOptimal(unittest.TestCase):
    """Construct emissions where the optimal path is deterministic."""

    def test_two_segments_clear_boundary(self):
        """Emission strongly favors tok0 in first half and tok1 in second half."""
        T, C = 20, 3
        tok0, tok1 = 0, 1
        lp = np.full((T, C), -100.0, dtype=np.float32)
        lp[:10, tok0] = 0.0
        lp[10:, tok1] = 0.0

        targets = [tok0, tok1]
        path = ctc_forced_align(lp, targets, blank=C, min_dur=1)

        self.assertTrue(
            np.all(path[:10] == 0),
            f"Expected state 0 in frames 0-9, got {path[:10]}"
        )
        self.assertTrue(
            np.all(path[10:] == 1),
            f"Expected state 1 in frames 10-19, got {path[10:]}"
        )

    def test_three_segments_equal_split(self):
        """Three equal segments; each target dominates its third."""
        T = 30
        C = 4
        targets = [0, 1, 2]
        lp = np.full((T, C), -100.0, dtype=np.float32)
        lp[:10, 0]   = 0.0
        lp[10:20, 1] = 0.0
        lp[20:, 2]   = 0.0

        path = ctc_forced_align(lp, targets, blank=C, min_dur=1)

        self.assertTrue(np.all(path[:10] == 0))
        self.assertTrue(np.all(path[10:20] == 1))
        self.assertTrue(np.all(path[20:] == 2))

    def test_output_dtype_and_shape(self):
        T, C = 15, 3
        targets = [0, 2]
        lp = _uniform_log_probs(T, C)
        path = ctc_forced_align(lp, targets, blank=C)
        self.assertEqual(path.shape, (T,))
        self.assertEqual(path.dtype, np.int16)


if __name__ == "__main__":
    unittest.main()
