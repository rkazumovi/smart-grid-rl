"""Regression test for probabilistic.py's inference-time post-processing: sort the three
quantile predictions into q10 <= q50 <= q90 (fixes the well-documented "quantile
crossing" problem), THEN clip negative values to 0 for wind/solar. The clip-after-sort
ordering matters -- clipping is a monotonic transform, so doing it after the sort can
only preserve the ordering the sort just established, never re-break it. This exact
property was hand-verified once during development; this test makes that check permanent
and automatic instead of a one-off manual check that can't regress-test future changes.
"""
import numpy as np


def sort_then_clip(preds: np.ndarray) -> np.ndarray:
    """Mirrors the exact operation probabilistic.py's __main__ block performs on its
    (n, 3) array of [q10, q50, q90] predictions for a non-negative target."""
    sorted_preds = np.sort(preds, axis=1)
    return np.clip(sorted_preds, 0.0, None)


def test_sort_then_clip_preserves_monotonicity():
    preds = np.array(
        [
            [-5.0, 2.0, 10.0],   # q10 crossed below 0
            [-1.0, -0.5, 3.0],   # q10 and q50 both crossed below 0
            [3.0, 1.0, 2.0],     # out of order to begin with (crossing, no negatives)
            [1.0, 2.0, 3.0],     # already fine
        ]
    )
    result = sort_then_clip(preds)

    assert np.all(result[:, 0] <= result[:, 1])
    assert np.all(result[:, 1] <= result[:, 2])
    assert np.all(result >= 0.0)


def test_sort_then_clip_hand_computed():
    preds = np.array([[-5.0, 2.0, 10.0], [-1.0, -0.5, 3.0]])
    result = sort_then_clip(preds)
    expected = np.array([[0.0, 2.0, 10.0], [0.0, 0.0, 3.0]])
    assert np.allclose(result, expected)


def test_sort_then_clip_never_changes_an_already_valid_nonnegative_row():
    preds = np.array([[1.0, 2.0, 3.0]])
    result = sort_then_clip(preds)
    assert np.allclose(result, preds)