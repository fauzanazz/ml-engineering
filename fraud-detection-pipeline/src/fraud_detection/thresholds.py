from dataclasses import dataclass

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

_DEFAULT_SWEEP = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95]


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
