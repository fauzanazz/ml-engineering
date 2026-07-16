from dataclasses import dataclass

import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score

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
    for threshold in thresholds:
        predictions = apply_threshold(scores, threshold=threshold)
        rows.append(
            ThresholdRow(
                threshold=threshold,
                precision=float(precision_score(labels, predictions, zero_division=0)),
                recall=float(recall_score(labels, predictions, zero_division=0)),
                f1=float(f1_score(labels, predictions, zero_division=0)),
                false_positives=int(((labels == 0) & (predictions == 1)).sum()),
                false_negatives=int(((labels == 1) & (predictions == 0)).sum()),
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
    if objective not in _VALID_OBJECTIVES:
        raise ValueError(f"objective must be one of {_VALID_OBJECTIVES}, got {objective!r}")
    if not (0.0 <= target_recall <= 1.0):
        raise ValueError(f"target_recall must satisfy 0 <= target_recall <= 1, got {target_recall}")

    if thresholds is not None:
        rows = sweep_thresholds(labels, scores, thresholds=thresholds)
        def f1_key(row: ThresholdRow) -> tuple[float, float]:
            return row.f1, row.threshold
        if objective == "f1":
            return max(rows, key=f1_key)
        qualifying = [row for row in rows if row.recall >= target_recall]
        return (
            max(qualifying, key=lambda row: (row.precision, row.recall, row.threshold))
            if qualifying
            else max(rows, key=f1_key)
        )

    precision, recall, candidates = precision_recall_curve(labels, scores)
    usable = (candidates > 0) & (candidates < 1)
    if not usable.any():
        raise ValueError("validation scores do not contain a usable threshold in (0, 1)")

    candidates = candidates[usable]
    precision = precision[:-1][usable]
    recall = recall[:-1][usable]
    denominators = precision + recall
    f1 = np.divide(2 * precision * recall, denominators, out=np.zeros_like(denominators), where=denominators != 0)

    if objective == "target-recall" and np.any(recall >= target_recall):
        eligible = np.flatnonzero(recall >= target_recall)
        winner = eligible[np.lexsort((candidates[eligible], recall[eligible], precision[eligible]))[-1]]
    else:
        winner = int(np.lexsort((candidates, f1))[-1])

    threshold = float(candidates[winner])
    predictions = apply_threshold(scores, threshold=threshold)
    return ThresholdRow(
        threshold=threshold,
        precision=float(precision[winner]),
        recall=float(recall[winner]),
        f1=float(f1[winner]),
        false_positives=int(((labels == 0) & (predictions == 1)).sum()),
        false_negatives=int(((labels == 1) & (predictions == 0)).sum()),
    )
