import numpy as np
import pytest

from fraud_detection.thresholds import (
    ThresholdRow,
    apply_threshold,
    sweep_thresholds,
    validate_threshold,
)


# --- validate_threshold ---


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.1, -10.0])
def test_validate_threshold_rejects_out_of_range(bad):
    with pytest.raises(ValueError):
        validate_threshold(bad)


def test_validate_threshold_accepts_valid_values():
    validate_threshold(0.5)
    validate_threshold(0.01)
    validate_threshold(0.99)


# --- apply_threshold ---


def test_apply_threshold_at_0_5_matches_default():
    scores = np.array([0.4, 0.6, 0.1, 0.9])
    result = apply_threshold(scores, threshold=0.5)
    expected = np.array([0, 1, 0, 1])
    np.testing.assert_array_equal(result, expected)


def test_lower_threshold_raises_recall_reduces_false_negatives():
    # 4 actual positives; at 0.5 two are missed (FN=2); at 0.2 none missed
    labels = np.array([1, 1, 1, 1, 0, 0])
    scores = np.array([0.45, 0.35, 0.8, 0.9, 0.1, 0.05])

    preds_high = apply_threshold(scores, threshold=0.5)
    preds_low = apply_threshold(scores, threshold=0.2)

    fn_high = int(((labels == 1) & (preds_high == 0)).sum())
    fn_low = int(((labels == 1) & (preds_low == 0)).sum())

    assert fn_low < fn_high


# --- sweep_thresholds ---


def test_sweep_thresholds_returns_row_per_threshold():
    labels = np.array([1, 1, 0, 0, 1])
    scores = np.array([0.9, 0.8, 0.2, 0.1, 0.4])

    rows = sweep_thresholds(labels, scores)

    assert len(rows) == 9  # default 9 thresholds
    assert all(isinstance(r, ThresholdRow) for r in rows)


def test_sweep_thresholds_includes_upper_thresholds():
    labels = np.array([1, 1, 0, 0, 1])
    scores = np.array([0.9, 0.8, 0.2, 0.1, 0.4])

    rows = sweep_thresholds(labels, scores)
    thresholds = [r.threshold for r in rows]

    assert 0.70 in thresholds
    assert 0.80 in thresholds
    assert 0.90 in thresholds
    assert 0.95 in thresholds


def test_sweep_thresholds_lower_threshold_higher_recall():
    labels = np.array([1, 1, 1, 0, 0])
    scores = np.array([0.45, 0.35, 0.25, 0.1, 0.05])

    rows = sweep_thresholds(labels, scores)
    row_by_threshold = {r.threshold: r for r in rows}

    # recall at 0.1 should be >= recall at 0.5
    assert row_by_threshold[0.10].recall >= row_by_threshold[0.50].recall


def test_sweep_thresholds_higher_threshold_higher_precision():
    labels = np.array([1, 1, 0, 0, 0])
    scores = np.array([0.96, 0.85, 0.72, 0.35, 0.15])

    rows = sweep_thresholds(labels, scores)
    row_by_threshold = {r.threshold: r for r in rows}

    # at 0.95 only the highest-confidence prediction fires → fewer FP → higher precision
    assert row_by_threshold[0.95].precision >= row_by_threshold[0.50].precision


def test_sweep_thresholds_row_has_required_fields():
    labels = np.array([1, 0])
    scores = np.array([0.9, 0.1])

    rows = sweep_thresholds(labels, scores)
    row = rows[0]

    assert hasattr(row, "threshold")
    assert hasattr(row, "precision")
    assert hasattr(row, "recall")
    assert hasattr(row, "f1")
    assert hasattr(row, "false_positives")
    assert hasattr(row, "false_negatives")
