import inspect

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score
from sklearn.model_selection import TimeSeriesSplit

from fraud_detection.features import FeaturePipeline
from fraud_detection.tuning import (
    _temporal_cv_scores,
    tune_lightgbm,
    tune_random_forest,
    tune_xgboost,
)


def _temporal_xy(n: int = 24) -> tuple[pd.DataFrame, pd.Series]:
    return (
        pd.DataFrame(
            {
                "Time": np.arange(n, dtype=float),
                "Amount": np.arange(1, n + 1, dtype=float),
                "V1": np.resize([0.1, 0.9], n),
            }
        ),
        pd.Series(np.resize([0, 1], n)),
    )


class _RecordingEstimator:
    def __init__(self, records: list[tuple[str, list[int]]]) -> None:
        self.records = records

    def fit(self, X, y):
        self.records.append(("fit", X.index.tolist()))
        return self

    def predict(self, X):
        return (X["V1"].to_numpy() >= 0.5).astype(int)

    def predict_proba(self, X):
        self.records.append(("score", X.index.tolist()))
        scores = X["V1"].to_numpy()
        return np.column_stack([1 - scores, scores])


def test_temporal_cv_is_forward_only_and_scores_average_precision():
    X, y = _temporal_xy()
    records: list[tuple[str, list[int]]] = []

    scores, effective = _temporal_cv_scores(
        X,
        y,
        n_splits=3,
        imbalance_strategy="none",
        build_estimator=lambda _: _RecordingEstimator(records),
    )

    assert effective == 3
    fits = [indices for kind, indices in records if kind == "fit"]
    validations = [indices for kind, indices in records if kind == "score"]
    assert all(max(train) < min(val) for train, val in zip(fits, validations, strict=True))
    expected = [average_precision_score(y.iloc[val], X.iloc[val]["V1"]) for val in validations]
    assert scores == pytest.approx(expected)


def test_temporal_cv_fits_a_fresh_pipeline_per_fold(monkeypatch):
    import fraud_detection.tuning as tuning

    X, y = _temporal_xy()
    fitted: list[tuple[int, float]] = []

    class RecordingPipeline(FeaturePipeline):
        def fit(self, frame):
            fitted.append((id(self), float(frame["Time"].max())))
            return super().fit(frame)

    monkeypatch.setattr(tuning, "FeaturePipeline", RecordingPipeline)
    _temporal_cv_scores(
        X,
        y,
        n_splits=3,
        imbalance_strategy="none",
        build_estimator=lambda _: _RecordingEstimator([]),
    )

    assert len(fitted) == 3
    assert len({identity for identity, _ in fitted}) == 3
    assert [time_max for _, time_max in fitted] == sorted(time_max for _, time_max in fitted)


def test_temporal_cv_uses_each_fold_training_weight():
    X, y = _temporal_xy()
    y.iloc[[1, 5, 9, 13, 17, 21]] = 0
    weights: list[float | None] = []

    _temporal_cv_scores(
        X,
        y,
        n_splits=3,
        imbalance_strategy="scale-pos-weight",
        build_estimator=lambda weight: weights.append(weight) or _RecordingEstimator([]),
    )

    expected = []
    for train, _ in TimeSeriesSplit(n_splits=3).split(X):
        fold_y = y.iloc[train]
        expected.append(float((fold_y == 0).sum() / (fold_y == 1).sum()))
    assert weights == pytest.approx(expected)


def test_temporal_cv_reduces_invalid_requested_fold_count():
    X, _ = _temporal_xy(12)
    y = pd.Series([0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1])

    _, effective = _temporal_cv_scores(
        X,
        y,
        n_splits=3,
        imbalance_strategy="none",
        build_estimator=lambda _: _RecordingEstimator([]),
    )

    assert effective == 2


def test_temporal_cv_fails_when_no_two_fold_evaluation_is_valid():
    X, _ = _temporal_xy(9)
    y = pd.Series([0, 0, 0, 0, 0, 0, 1, 0, 1])

    with pytest.raises(ValueError, match="no valid temporal CV"):
        _temporal_cv_scores(
            X,
            y,
            n_splits=3,
            imbalance_strategy="none",
            build_estimator=lambda _: _RecordingEstimator([]),
        )


def test_temporal_cv_requires_chronological_input():
    X, y = _temporal_xy()
    X.loc[5, "Time"] = -1

    with pytest.raises(ValueError, match="monotonic nondecreasing"):
        _temporal_cv_scores(
            X,
            y,
            n_splits=2,
            imbalance_strategy="none",
            build_estimator=lambda _: _RecordingEstimator([]),
        )


@pytest.mark.parametrize("tune", [tune_lightgbm, tune_random_forest, tune_xgboost])
def test_tuning_functions_share_the_public_signature(tune):
    signature = inspect.signature(tune)
    assert list(signature.parameters) == [
        "X",
        "y",
        "n_iter",
        "cv",
        "scoring",
        "random_state",
        "imbalance_strategy",
    ]
    assert signature.parameters["scoring"].default == "average_precision"
    assert signature.parameters["cv"].default == 3


@pytest.mark.parametrize("tune", [tune_lightgbm, tune_random_forest, tune_xgboost])
def test_tuning_rejects_non_average_precision_scoring(tune):
    X, y = _temporal_xy()
    with pytest.raises(ValueError, match="average_precision"):
        tune(X, y, n_iter=1, cv=2, scoring="roc_auc")


@pytest.mark.parametrize("tune", [tune_lightgbm, tune_random_forest, tune_xgboost])
def test_tuning_rejects_invalid_controls(tune):
    X, y = _temporal_xy()
    with pytest.raises(ValueError, match="n_iter"):
        tune(X, y, n_iter=0, cv=2)
    with pytest.raises(ValueError, match="cv"):
        tune(X, y, n_iter=1, cv=1)


def test_random_forest_tuning_is_seeded_and_reports_temporal_metadata():
    X, y = _temporal_xy()
    first = tune_random_forest(
        X,
        y,
        n_iter=2,
        cv=3,
        random_state=7,
        imbalance_strategy="scale-pos-weight",
    )
    second = tune_random_forest(
        X,
        y,
        n_iter=2,
        cv=3,
        random_state=7,
        imbalance_strategy="scale-pos-weight",
    )

    assert first.best_params == second.best_params
    assert first.best_score == pytest.approx(second.best_score)
    assert first.scoring == "average_precision"
    assert first.cv_strategy == "TimeSeriesSplit"
    assert first.cv_splits == 3
    assert first.random_state == 7
    model = first.best_factory.create(scale_pos_weight=4.0)
    assert model.random_state == 7
    assert model.class_weight == {0: 1.0, 1: 4.0}
