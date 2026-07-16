import math
import warnings
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class ClassificationMetrics:
    precision: float
    recall: float
    f1: float
    pr_auc: float
    roc_auc: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    positive_support: int
    negative_support: int


class MetricsAdapter(Protocol):
    def compute(
        self,
        labels: np.ndarray,
        *,
        predictions: np.ndarray,
        scores: np.ndarray,
    ) -> ClassificationMetrics: ...


def _has_multiple_classes(labels: np.ndarray) -> bool:
    return len(np.unique(labels)) > 1


class SklearnMetricsAdapter:
    def compute(
        self,
        labels: np.ndarray,
        *,
        predictions: np.ndarray,
        scores: np.ndarray,
    ) -> ClassificationMetrics:
        has_multiple_classes = _has_multiple_classes(labels)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            pr_auc = float(average_precision_score(labels, scores))
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        return ClassificationMetrics(
            precision=float(precision_score(labels, predictions, zero_division=0)),
            recall=float(recall_score(labels, predictions, zero_division=0)),
            f1=float(f1_score(labels, predictions, zero_division=0)),
            pr_auc=pr_auc,
            roc_auc=float(roc_auc_score(labels, scores)) if has_multiple_classes else math.nan,
            true_positives=int(tp),
            true_negatives=int(tn),
            false_positives=int(fp),
            false_negatives=int(fn),
            positive_support=int((labels == 1).sum()),
            negative_support=int((labels == 0).sum()),
        )
