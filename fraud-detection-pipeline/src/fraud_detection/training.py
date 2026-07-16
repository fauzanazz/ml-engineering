import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from fraud_detection.data import load_three_way_split, load_time_split_batch
from fraud_detection.features import FeaturePipeline
from fraud_detection.metrics import ClassificationMetrics, MetricsAdapter, SklearnMetricsAdapter
from fraud_detection.models import ModelFactory
from fraud_detection.thresholds import apply_threshold, select_threshold_on_validation, validate_threshold

ImbalanceStrategy = Literal["none", "scale-pos-weight"]
ThresholdObjective = Literal["f1", "target-recall"]


@dataclass(frozen=True)
class PartitionAudit:
    rows: int
    positives: int
    negatives: int
    time_min: float
    time_max: float


@dataclass(frozen=True)
class SplitAudit:
    train: PartitionAudit
    test: PartitionAudit
    val: PartitionAudit | None = None


@dataclass(frozen=True)
class TrainingResult:
    predictions: list[int]
    training_accuracy: float
    test_accuracy: float
    metrics: ClassificationMetrics
    model: Any = field(compare=False)
    feature_pipeline: FeaturePipeline = field(compare=False)
    input_columns: tuple[str, ...]
    effective_threshold: float
    threshold_target_met: bool | None
    threshold_fallback_used: bool
    split_audit: SplitAudit
    test_labels: np.ndarray = field(default_factory=lambda: np.array([]), compare=False)
    test_scores: np.ndarray = field(default_factory=lambda: np.array([]), compare=False)
    val_threshold: float | None = field(default=None, compare=False)
    val_metrics: ClassificationMetrics | None = field(default=None, compare=False)
    threshold_objective: ThresholdObjective | None = field(default=None, compare=False)
    target_recall: float | None = field(default=None, compare=False)
    predict_proba_latency_s: float | None = field(default=None, compare=False)
    predict_proba_latency_per_row_s: float | None = field(default=None, compare=False)
    single_row_latency_s: float | None = field(default=None, compare=False)


def _measure_single_row_latency(model, one_row, *, n_repeats: int = 5) -> float | None:
    if not hasattr(model, "predict_proba"):
        return None
    model.predict_proba(one_row)
    times = []
    for _ in range(n_repeats):
        started = time.perf_counter()
        model.predict_proba(one_row)
        times.append(time.perf_counter() - started)
    return float(np.median(times))


def compute_scale_pos_weight(train_target: pd.Series) -> float:
    positive_count = int((train_target == 1).sum())
    negative_count = int((train_target == 0).sum())
    if positive_count == 0:
        raise ValueError("compute_scale_pos_weight: no positive samples in train target")
    if negative_count == 0:
        raise ValueError("compute_scale_pos_weight: no negative samples in train target")
    return negative_count / positive_count


def _audit_partition(name: str, features: pd.DataFrame, target: pd.Series) -> PartitionAudit:
    if features.empty:
        raise ValueError(f"{name} split is empty")
    return PartitionAudit(
        rows=len(features),
        positives=int((target == 1).sum()),
        negatives=int((target == 0).sum()),
        time_min=float(features["Time"].min()),
        time_max=float(features["Time"].max()),
    )


def _fit_and_transform(
    raw_train: pd.DataFrame,
    *partitions: pd.DataFrame,
) -> tuple[FeaturePipeline, list[pd.DataFrame]]:
    pipeline = FeaturePipeline().fit(raw_train)
    return pipeline, [pipeline.transform(partition) for partition in (raw_train, *partitions)]


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
    validate_threshold(decision_threshold)
    adapter = metrics_adapter if metrics_adapter is not None else SklearnMetricsAdapter()

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
            metrics_adapter=adapter,
            imbalance_strategy=imbalance_strategy,
        )

    raw_train, raw_test, train_target, test_target = load_time_split_batch(
        data_path, batch_size, target_column, test_size
    )
    split_audit = SplitAudit(
        train=_audit_partition("train", raw_train, train_target),
        test=_audit_partition("test", raw_test, test_target),
    )
    pipeline, (train_features, test_features) = _fit_and_transform(raw_train, raw_test)

    weight = compute_scale_pos_weight(train_target) if imbalance_strategy == "scale-pos-weight" else None
    model = model_factory.create(scale_pos_weight=weight)
    model.fit(train_features, train_target)
    train_predictions = model.predict(train_features)
    started = time.perf_counter()
    test_scores = model.predict_proba(test_features)[:, 1]
    batch_latency = time.perf_counter() - started
    test_labels = test_target.to_numpy()
    test_predictions = apply_threshold(test_scores, threshold=decision_threshold)

    return _build_result(
        model=model,
        pipeline=pipeline,
        input_columns=tuple(raw_train.columns),
        split_audit=split_audit,
        train_target=train_target,
        train_predictions=train_predictions,
        test_features=test_features,
        test_labels=test_labels,
        test_scores=test_scores,
        test_predictions=test_predictions,
        metrics_adapter=adapter,
        effective_threshold=decision_threshold,
        threshold_target_met=None,
        threshold_fallback_used=False,
        batch_latency=batch_latency,
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
    raw_train, raw_val, raw_test, train_target, val_target, test_target = load_three_way_split(
        path=data_path,
        val_size=val_size,
        test_size=test_size,
        target_column=target_column,
        batch_size=batch_size,
    )
    split_audit = SplitAudit(
        train=_audit_partition("train", raw_train, train_target),
        val=_audit_partition("validation", raw_val, val_target),
        test=_audit_partition("test", raw_test, test_target),
    )
    pipeline, (train_features, val_features, test_features) = _fit_and_transform(
        raw_train, raw_val, raw_test
    )

    weight = compute_scale_pos_weight(train_target) if imbalance_strategy == "scale-pos-weight" else None
    model = model_factory.create(scale_pos_weight=weight)
    model.fit(train_features, train_target)

    val_labels = val_target.to_numpy()
    val_scores = model.predict_proba(val_features)[:, 1]
    requested_recall = target_recall if target_recall is not None else 0.95
    selected = select_threshold_on_validation(
        val_labels,
        val_scores,
        objective=threshold_objective,
        target_recall=requested_recall,
    )
    val_predictions = apply_threshold(val_scores, threshold=selected.threshold)
    val_metrics = metrics_adapter.compute(val_labels, predictions=val_predictions, scores=val_scores)
    target_met = selected.recall >= requested_recall if threshold_objective == "target-recall" else None

    train_predictions = model.predict(train_features)
    test_labels = test_target.to_numpy()
    started = time.perf_counter()
    test_scores = model.predict_proba(test_features)[:, 1]
    batch_latency = time.perf_counter() - started
    test_predictions = apply_threshold(test_scores, threshold=selected.threshold)

    return _build_result(
        model=model,
        pipeline=pipeline,
        input_columns=tuple(raw_train.columns),
        split_audit=split_audit,
        train_target=train_target,
        train_predictions=train_predictions,
        test_features=test_features,
        test_labels=test_labels,
        test_scores=test_scores,
        test_predictions=test_predictions,
        metrics_adapter=metrics_adapter,
        effective_threshold=selected.threshold,
        threshold_target_met=target_met,
        threshold_fallback_used=target_met is False,
        batch_latency=batch_latency,
        val_threshold=selected.threshold,
        val_metrics=val_metrics,
        threshold_objective=threshold_objective,
        target_recall=target_recall,
    )


def _build_result(
    *,
    model: Any,
    pipeline: FeaturePipeline,
    input_columns: tuple[str, ...],
    split_audit: SplitAudit,
    train_target: pd.Series,
    train_predictions: np.ndarray,
    test_features: pd.DataFrame,
    test_labels: np.ndarray,
    test_scores: np.ndarray,
    test_predictions: np.ndarray,
    metrics_adapter: MetricsAdapter,
    effective_threshold: float,
    threshold_target_met: bool | None,
    threshold_fallback_used: bool,
    batch_latency: float,
    val_threshold: float | None = None,
    val_metrics: ClassificationMetrics | None = None,
    threshold_objective: ThresholdObjective | None = None,
    target_recall: float | None = None,
) -> TrainingResult:
    row_count = len(test_labels)
    return TrainingResult(
        predictions=test_predictions.tolist(),
        training_accuracy=float((train_predictions == train_target).mean()),
        test_accuracy=float((test_predictions == test_labels).mean()),
        metrics=metrics_adapter.compute(test_labels, predictions=test_predictions, scores=test_scores),
        model=model,
        feature_pipeline=pipeline,
        input_columns=input_columns,
        effective_threshold=effective_threshold,
        threshold_target_met=threshold_target_met,
        threshold_fallback_used=threshold_fallback_used,
        split_audit=split_audit,
        test_labels=test_labels,
        test_scores=test_scores,
        val_threshold=val_threshold,
        val_metrics=val_metrics,
        threshold_objective=threshold_objective,
        target_recall=target_recall,
        predict_proba_latency_s=batch_latency,
        predict_proba_latency_per_row_s=batch_latency / row_count,
        single_row_latency_s=_measure_single_row_latency(model, test_features.iloc[:1]),
    )
