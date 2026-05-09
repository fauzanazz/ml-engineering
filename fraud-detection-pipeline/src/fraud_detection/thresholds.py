from dataclasses import dataclass

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

_DEFAULT_SWEEP = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95]
_VALID_OBJECTIVES = {"f1", "target-recall"}


@dataclass(frozen=True)
class ThresholdRow:
    threshold: float
    precision: float
    recall: float
    f1: float
    false_positives: int
    false_negatives: int


def validate_threshold(threshold: float) -> float:
    if not (0 < threshold < 1):
        raise ValueError(f"decision_threshold must satisfy 0 < threshold < 1, got {threshold}")
    return threshold


def apply_threshold(scores: np.ndarray, *, threshold: float) -> np.ndarray:
    return (scores >= threshold).astype(int)


def sweep_thresholds(
    labels: np.ndarray,
    scores: np.ndarray,
    thresholds: list[float] = _DEFAULT_SWEEP,
) -> list[ThresholdRow]:
    rows = []
    for t in thresholds:
        preds = apply_threshold(scores, threshold=t)
        fp = int(((labels == 0) & (preds == 1)).sum())
        fn = int(((labels == 1) & (preds == 0)).sum())
        rows.append(
            ThresholdRow(
                threshold=t,
                precision=float(precision_score(labels, preds, zero_division=0)),
                recall=float(recall_score(labels, preds, zero_division=0)),
                f1=float(f1_score(labels, preds, zero_division=0)),
                false_positives=fp,
                false_negatives=fn,
            )
        )
    return rows


def select_threshold_on_validation(
    labels: np.ndarray,
    scores: np.ndarray,
    objective: str = "f1",
    target_recall: float = 0.95,
    thresholds: list[float] | None = None,
) -> ThresholdRow:
    """Select best threshold from validation sweep.

    objective='f1'           → row with highest F1; tie-break: higher threshold.
    objective='target-recall' → among rows with recall >= target_recall, pick
                                highest precision; tie-break: higher recall,
                                then higher threshold.
                                Fallback when none qualify: max F1 row
                                (same tie-break as f1 mode).

    Args:
        labels: ground-truth binary labels.
        scores: predicted probabilities.
        objective: 'f1' or 'target-recall'.
        target_recall: required recall floor; only used when objective='target-recall'.
        thresholds: custom threshold list; defaults to _DEFAULT_SWEEP.

    Returns:
        ThresholdRow with selected threshold and its metrics.

    Raises:
        ValueError: objective not in {'f1','target-recall'} or target_recall not in [0,1].
    """
    if objective not in _VALID_OBJECTIVES:
        raise ValueError(f"objective must be one of {_VALID_OBJECTIVES}, got {objective!r}")
    if not (0.0 <= target_recall <= 1.0):
        raise ValueError(f"target_recall must satisfy 0 <= target_recall <= 1, got {target_recall}")

    rows = sweep_thresholds(labels, scores, thresholds=thresholds or _DEFAULT_SWEEP)

    def _f1_key(row: ThresholdRow) -> tuple:
        return (row.f1, row.threshold)

    if objective == "f1":
        return max(rows, key=_f1_key)

    # objective == 'target-recall'
    qualifying = [r for r in rows if r.recall >= target_recall]
    if qualifying:
        return max(qualifying, key=lambda r: (r.precision, r.recall, r.threshold))

    # fallback: no threshold meets target — pick max F1 (documented behaviour)
    return max(rows, key=_f1_key)
