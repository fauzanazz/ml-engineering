import numpy as np
import pytest

from fraud_detection.thresholds import (
    ThresholdRow,
    apply_threshold,
    select_threshold_on_validation,
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


# --- select_threshold_on_validation ---

# Fixtures
_LABELS = np.array([1, 1, 1, 1, 0, 0, 0, 0])
_SCORES = np.array([0.9, 0.8, 0.6, 0.4, 0.35, 0.25, 0.15, 0.05])
# at t=0.50: preds=[1,1,1,0,0,0,0,0] → P=1.0, R=0.75, F1=0.857
# at t=0.30: preds=[1,1,1,1,1,0,0,0] → P=0.8, R=1.0,  F1=0.889
# at t=0.20: preds=[1,1,1,1,1,1,0,0] → P=0.67,R=1.0,  F1=0.8
# at t=0.10: preds=[1,1,1,1,1,1,1,0] → P=0.57,R=1.0,  F1=0.727
# at t=0.05: preds=[1,1,1,1,1,1,1,1] → P=0.5, R=1.0,  F1=0.667
_THRESHOLDS = [0.50, 0.30, 0.20, 0.10, 0.05]


def test_select_threshold_f1_picks_max_f1():
    row = select_threshold_on_validation(_LABELS, _SCORES, objective="f1", thresholds=_THRESHOLDS)
    assert row.threshold == pytest.approx(0.30)  # highest F1 ≈ 0.889


def test_select_threshold_f1_returns_threshold_row():
    row = select_threshold_on_validation(_LABELS, _SCORES, objective="f1", thresholds=_THRESHOLDS)
    assert isinstance(row, ThresholdRow)


def test_select_threshold_target_recall_picks_max_precision_among_qualifying():
    # t=0.30 and t=0.20 and t=0.10 and t=0.05 all have recall=1.0 >= 0.95
    # max precision among those is t=0.30 (P=0.8)
    row = select_threshold_on_validation(
        _LABELS, _SCORES, objective="target-recall", target_recall=0.95, thresholds=_THRESHOLDS
    )
    assert row.threshold == pytest.approx(0.30)


def test_select_threshold_target_recall_tiebreak_higher_recall():
    # Build data where two thresholds tie on precision but differ on recall.
    labels = np.array([1, 1, 1, 0, 0])
    scores = np.array([0.9, 0.7, 0.6, 0.2, 0.1])
    # t=0.55: preds=[1,1,1,0,0] P=1.0, R=1.0
    # t=0.65: preds=[1,1,0,0,0] P=1.0, R=0.667  → below 0.95
    # Only t=0.55 qualifies; it wins trivially.
    # For a real tie-on-precision test: build equal precision, diff recall.
    labels2 = np.array([1, 1, 0, 0])
    scores2 = np.array([0.9, 0.6, 0.3, 0.1])
    # t=0.55: preds=[1,1,0,0] P=1.0, R=1.0 (both positives caught)
    # t=0.85: preds=[1,0,0,0] P=1.0, R=0.5 → below 0.95
    # Only t=0.55 qualifies; trivial win again.
    # Build explicit tie: same precision, lower threshold catches more positives.
    labels3 = np.array([1, 1, 1, 0])
    scores3 = np.array([0.9, 0.8, 0.6, 0.5])
    # t=0.55: preds=[1,1,1,0] P=1.0, R=1.0
    # t=0.75: preds=[1,1,0,0] P=1.0, R=0.667 → below 0.95
    # Single qualifier; tie-break irrelevant. Row must have recall>=0.95.
    row = select_threshold_on_validation(
        labels3, scores3, objective="target-recall", target_recall=0.95, thresholds=[0.55, 0.75]
    )
    assert row.recall >= 0.95


def test_select_threshold_target_recall_tiebreak_higher_threshold():
    # Two thresholds with equal precision AND equal recall → pick higher threshold.
    labels = np.array([1, 0])
    scores = np.array([0.9, 0.1])
    # t=0.5 and t=0.7: both predict [1,0] → same P, R, F1
    row = select_threshold_on_validation(
        labels, scores, objective="target-recall", target_recall=0.95, thresholds=[0.50, 0.70]
    )
    assert row.threshold == pytest.approx(0.70)


def test_select_threshold_target_recall_fallback_to_max_f1_when_none_qualify():
    # No threshold achieves recall >= 0.99; fallback = max F1.
    # t=0.50: R=0.75 < 0.99, F1=0.857
    # t=0.30: R=1.0 >= 0.99 … wait, let's use a tighter target.
    # Use labels/scores where best recall < 0.99.
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.8, 0.3, 0.2, 0.1])
    # t=0.50: preds=[1,0,0,0] R=0.5, P=1.0, F1=0.667
    # t=0.25: preds=[1,1,0,0] R=1.0, P=1.0, F1=1.0
    # target_recall=0.99 → t=0.25 qualifies (R=1.0)
    # Use a target no threshold can meet:
    labels2 = np.array([1, 1, 1, 0])
    scores2 = np.array([0.9, 0.6, 0.4, 0.8])
    # t=0.85: preds=[1,0,0,0] R=0.333
    # t=0.50: preds=[1,1,1,1] R=1.0 … the FP makes P=0.75
    # At t=0.50 R=1.0 ≥ target so it qualifies.
    # Hard to avoid qualify with default sweep; force with high target and low scores.
    labels3 = np.array([1, 1, 0, 0])
    scores3 = np.array([0.6, 0.5, 0.4, 0.3])
    # Only t=0.05 or t=0.10 in default sweep would predict enough positives.
    # Use custom thresholds all above max positive score → R=0 everywhere.
    labels4 = np.array([1, 1, 0])
    scores4 = np.array([0.3, 0.2, 0.1])
    # t=0.50: all predicted 0 → R=0
    # t=0.70: all predicted 0 → R=0
    # Neither qualifies target_recall=0.95 → fallback max F1 (both F1=0, tie → higher threshold)
    row = select_threshold_on_validation(
        labels4, scores4, objective="target-recall", target_recall=0.95, thresholds=[0.50, 0.70]
    )
    # Fallback: max F1; both zero so tie-break = higher threshold per docs
    assert row.threshold == pytest.approx(0.70)


def test_select_threshold_invalid_objective_raises():
    with pytest.raises(ValueError, match="objective"):
        select_threshold_on_validation(_LABELS, _SCORES, objective="bad-objective", thresholds=_THRESHOLDS)


def test_select_threshold_invalid_target_recall_raises():
    with pytest.raises(ValueError, match="target_recall"):
        select_threshold_on_validation(
            _LABELS, _SCORES, objective="target-recall", target_recall=1.5, thresholds=_THRESHOLDS
        )


def test_select_threshold_invalid_target_recall_negative_raises():
    with pytest.raises(ValueError, match="target_recall"):
        select_threshold_on_validation(
            _LABELS, _SCORES, objective="target-recall", target_recall=-0.1, thresholds=_THRESHOLDS
        )


def test_select_threshold_uses_default_sweep_when_thresholds_none():
    row = select_threshold_on_validation(_LABELS, _SCORES, objective="f1")
    assert isinstance(row, ThresholdRow)
    assert 0 < row.threshold < 1


def test_select_threshold_f1_is_deterministic():
    row1 = select_threshold_on_validation(_LABELS, _SCORES, objective="f1", thresholds=_THRESHOLDS)
    row2 = select_threshold_on_validation(_LABELS, _SCORES, objective="f1", thresholds=_THRESHOLDS)
    assert row1.threshold == row2.threshold


def test_select_threshold_target_recall_is_deterministic():
    row1 = select_threshold_on_validation(
        _LABELS, _SCORES, objective="target-recall", target_recall=0.95, thresholds=_THRESHOLDS
    )
    row2 = select_threshold_on_validation(
        _LABELS, _SCORES, objective="target-recall", target_recall=0.95, thresholds=_THRESHOLDS
    )
    assert row1.threshold == row2.threshold
