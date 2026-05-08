from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from fraud_detection.data import load_time_split_batch
from fraud_detection.metrics import ClassificationMetrics, MetricsAdapter, SklearnMetricsAdapter
from fraud_detection.models import ModelFactory

ImbalanceStrategy = Literal["none", "scale-pos-weight"]


@dataclass(frozen=True)
class TrainingResult:
    predictions: list[int]
    training_accuracy: float
    test_accuracy: float
    metrics: ClassificationMetrics
    test_labels: np.ndarray = field(default_factory=lambda: np.array([]), compare=False)
    test_scores: np.ndarray = field(default_factory=lambda: np.array([]), compare=False)
    model: Any = field(default=None, compare=False)


def compute_scale_pos_weight(train_target: pd.Series) -> float:
    positive_count = int((train_target == 1).sum())
    negative_count = int((train_target == 0).sum())
    if positive_count == 0:
        raise ValueError("compute_scale_pos_weight: no positive samples in train target")
    if negative_count == 0:
        raise ValueError("compute_scale_pos_weight: no negative samples in train target")
    return negative_count / positive_count


def train_one_batch(
    data_path: Path | str,
    model_factory: ModelFactory,
    batch_size: int,
    target_column: str = "Class",
    test_size: float = 0.2,
    metrics_adapter: MetricsAdapter | None = None,
    imbalance_strategy: ImbalanceStrategy = "none",
    decision_threshold: float = 0.5,
) -> TrainingResult:
    from fraud_detection.thresholds import apply_threshold, validate_threshold

    validate_threshold(decision_threshold)
    resolved_adapter = metrics_adapter if metrics_adapter is not None else SklearnMetricsAdapter()
    train_features, test_features, train_target, test_target = load_time_split_batch(
        data_path,
        batch_size,
        target_column,
        test_size,
    )

    scale_pos_weight = compute_scale_pos_weight(train_target) if imbalance_strategy == "scale-pos-weight" else None
    model = model_factory.create(scale_pos_weight=scale_pos_weight)
    model.fit(train_features, train_target)
    train_predictions = model.predict(train_features)
    test_scores = model.predict_proba(test_features)[:, 1]
    test_predictions = apply_threshold(test_scores, threshold=decision_threshold)

    test_labels = test_target.to_numpy()
    training_accuracy = float((train_predictions == train_target).mean())
    test_accuracy = float((test_predictions == test_labels).mean())
    metrics = resolved_adapter.compute(
        test_labels,
        predictions=test_predictions,
        scores=test_scores,
    )

    return TrainingResult(
        predictions=test_predictions.tolist(),
        training_accuracy=training_accuracy,
        test_accuracy=test_accuracy,
        metrics=metrics,
        test_labels=test_labels,
        test_scores=test_scores,
        model=model,
    )
