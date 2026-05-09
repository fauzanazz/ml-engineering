import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from fraud_detection.data import load_time_split_batch
from fraud_detection.metrics import ClassificationMetrics, MetricsAdapter, SklearnMetricsAdapter
from fraud_detection.models import ModelFactory

ImbalanceStrategy = Literal["none", "scale-pos-weight"]
ThresholdObjective = Literal["f1", "target-recall"]


@dataclass(frozen=True)
class SplitCounts:
    train: int
    test: int
    val: int | None = None


@dataclass(frozen=True)
class TrainingResult:
    predictions: list[int]
    training_accuracy: float
    test_accuracy: float
    metrics: ClassificationMetrics
    test_labels: np.ndarray = field(default_factory=lambda: np.array([]), compare=False)
    test_scores: np.ndarray = field(default_factory=lambda: np.array([]), compare=False)
    model: Any = field(default=None, compare=False)
    val_threshold: float | None = field(default=None, compare=False)
    val_metrics: ClassificationMetrics | None = field(default=None, compare=False)
    threshold_objective: ThresholdObjective | None = field(default=None, compare=False)
    target_recall: float | None = field(default=None, compare=False)
    split_counts: SplitCounts | None = field(default=None, compare=False)
    predict_proba_latency_s: float | None = field(default=None, compare=False)
    predict_proba_latency_per_row_s: float | None = field(default=None, compare=False)
    single_row_latency_s: float | None = field(default=None, compare=False)

    @property
    def inference_latency_s(self) -> float | None:
        """Alias for predict_proba_latency_s — backward compat."""
        return self.predict_proba_latency_s


def _measure_single_row_latency(model, one_row, *, n_repeats: int = 5) -> float | None:
    """Median latency for a single-row predict_proba call.

    One warmup call runs first (excluded from timing) so JIT/cache effects
    don't skew the first timed sample.

    Returns None if model lacks predict_proba.
    """
    if not hasattr(model, "predict_proba"):
        return None
    model.predict_proba(one_row)  # warmup — excluded from timing
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        model.predict_proba(one_row)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


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
    val_size: float | None = None,
    threshold_objective: ThresholdObjective = "f1",
    target_recall: float | None = None,
) -> TrainingResult:
    from fraud_detection.thresholds import apply_threshold, select_threshold_on_validation, validate_threshold

    validate_threshold(decision_threshold)
    resolved_adapter = metrics_adapter if metrics_adapter is not None else SklearnMetricsAdapter()

    if val_size is not None:
        return _train_with_validation(
            data_path=data_path,
            model_factory=model_factory,
            batch_size=batch_size,
            target_column=target_column,
            test_size=test_size,
            val_size=val_size,
            threshold_objective=threshold_objective,
            target_recall=target_recall,
            metrics_adapter=resolved_adapter,
            imbalance_strategy=imbalance_strategy,
        )

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
    _t0 = time.perf_counter()
    test_scores = model.predict_proba(test_features)[:, 1]
    predict_proba_latency_s = time.perf_counter() - _t0
    test_predictions = apply_threshold(test_scores, threshold=decision_threshold)

    test_labels = test_target.to_numpy()
    training_accuracy = float((train_predictions == train_target).mean())
    test_accuracy = float((test_predictions == test_labels).mean())
    metrics = resolved_adapter.compute(
        test_labels,
        predictions=test_predictions,
        scores=test_scores,
    )
    n_test_rows = len(test_labels)
    predict_proba_latency_per_row_s = predict_proba_latency_s / n_test_rows if n_test_rows > 0 else None
    single_row_latency_s = _measure_single_row_latency(model, test_features.iloc[:1])

    return TrainingResult(
        predictions=test_predictions.tolist(),
        training_accuracy=training_accuracy,
        test_accuracy=test_accuracy,
        metrics=metrics,
        test_labels=test_labels,
        test_scores=test_scores,
        model=model,
        split_counts=SplitCounts(train=len(train_target), test=len(test_target)),
        predict_proba_latency_s=predict_proba_latency_s,
        predict_proba_latency_per_row_s=predict_proba_latency_per_row_s,
        single_row_latency_s=single_row_latency_s,
    )


def _train_with_validation(
    data_path: Path | str,
    model_factory: ModelFactory,
    batch_size: int,
    target_column: str,
    test_size: float,
    val_size: float,
    threshold_objective: ThresholdObjective,
    target_recall: float | None,
    metrics_adapter: MetricsAdapter,
    imbalance_strategy: ImbalanceStrategy,
) -> TrainingResult:
    from fraud_detection.data import load_three_way_split
    from fraud_detection.features import FeaturePipeline
    from fraud_detection.thresholds import apply_threshold, select_threshold_on_validation

    raw_train_feat, raw_val_feat, raw_test_feat, train_target, val_target, test_target = (
        load_three_way_split(
            path=data_path,
            val_size=val_size,
            test_size=test_size,
            target_column=target_column,
            batch_size=batch_size,
        )
    )

    pipeline = FeaturePipeline().fit(raw_train_feat)
    train_features = pipeline.transform(raw_train_feat)
    val_features = pipeline.transform(raw_val_feat)
    test_features = pipeline.transform(raw_test_feat)

    scale_pos_weight = compute_scale_pos_weight(train_target) if imbalance_strategy == "scale-pos-weight" else None
    model = model_factory.create(scale_pos_weight=scale_pos_weight)
    model.fit(train_features, train_target)

    val_labels = val_target.to_numpy()
    val_scores = model.predict_proba(val_features)[:, 1]
    best_row = select_threshold_on_validation(
        val_labels,
        val_scores,
        objective=threshold_objective,
        target_recall=target_recall if target_recall is not None else 0.95,
    )
    val_threshold = best_row.threshold

    val_predictions = apply_threshold(val_scores, threshold=val_threshold)
    val_metrics = metrics_adapter.compute(val_labels, predictions=val_predictions, scores=val_scores)

    train_predictions = model.predict(train_features)
    test_labels = test_target.to_numpy()
    _t0 = time.perf_counter()
    test_scores = model.predict_proba(test_features)[:, 1]
    predict_proba_latency_s = time.perf_counter() - _t0
    test_predictions = apply_threshold(test_scores, threshold=val_threshold)

    training_accuracy = float((train_predictions == train_target).mean())
    test_accuracy = float((test_predictions == test_labels).mean())
    metrics = metrics_adapter.compute(test_labels, predictions=test_predictions, scores=test_scores)
    n_test_rows = len(test_labels)
    predict_proba_latency_per_row_s = predict_proba_latency_s / n_test_rows if n_test_rows > 0 else None
    single_row_latency_s = _measure_single_row_latency(model, test_features.iloc[:1])

    return TrainingResult(
        predictions=test_predictions.tolist(),
        training_accuracy=training_accuracy,
        test_accuracy=test_accuracy,
        metrics=metrics,
        test_labels=test_labels,
        test_scores=test_scores,
        model=model,
        val_threshold=val_threshold,
        val_metrics=val_metrics,
        threshold_objective=threshold_objective,
        target_recall=target_recall,
        split_counts=SplitCounts(train=len(train_target), val=len(val_target), test=len(test_target)),
        predict_proba_latency_s=predict_proba_latency_s,
        predict_proba_latency_per_row_s=predict_proba_latency_per_row_s,
        single_row_latency_s=single_row_latency_s,
    )
