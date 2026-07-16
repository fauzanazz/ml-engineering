import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import TimeSeriesSplit

from fraud_detection.features import FeaturePipeline
from fraud_detection.models import Classifier, ModelFactory
from fraud_detection.training import ImbalanceStrategy, compute_scale_pos_weight

optuna.logging.set_verbosity(logging.WARNING)

_RF_PARAM_SPACE = {
    "n_estimators": ([50, 100, 200], "categorical"),
    "max_depth": ([None, 5, 10, 20], "categorical"),
    "min_samples_split": ([2, 5, 10], "categorical"),
    "min_samples_leaf": ([1, 2, 4], "categorical"),
}
_XGB_PARAM_SPACE = {
    "n_estimators": ([50, 100, 200], "categorical"),
    "max_depth": ([3, 5, 7, 9], "categorical"),
    "learning_rate": ([0.01, 0.05, 0.1, 0.2, 0.3], "categorical"),
    "subsample": ([0.6, 0.8, 1.0], "categorical"),
    "colsample_bytree": ([0.6, 0.8, 1.0], "categorical"),
}
_LGBM_PARAM_SPACE = {
    "n_estimators": ([50, 100, 200], "categorical"),
    "max_depth": ([-1, 3, 5, 7], "categorical"),
    "learning_rate": ([0.01, 0.05, 0.1, 0.2, 0.3], "categorical"),
    "num_leaves": ([15, 31, 63, 127], "categorical"),
    "subsample": ([0.6, 0.8, 1.0], "categorical"),
}


def _validate_tuning_inputs(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_iter: int,
    cv: int,
    scoring: str,
) -> None:
    if n_iter < 1:
        raise ValueError(f"n_iter must be >= 1, got {n_iter}")
    if cv < 2:
        raise ValueError(f"cv must be >= 2, got {cv}")
    if scoring != "average_precision":
        raise ValueError("scoring must be 'average_precision'")
    if "Time" not in X.columns:
        raise ValueError("X must contain Time for temporal CV")
    if not X["Time"].is_monotonic_increasing:
        raise ValueError("X.Time must be monotonic nondecreasing for temporal CV")
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows")


def _temporal_cv_scores(
    X,
    y,
    *,
    n_splits: int,
    imbalance_strategy: ImbalanceStrategy,
    build_estimator: Callable[[float | None], Classifier],
) -> tuple[list[float], int]:
    if "Time" not in X.columns or not X["Time"].is_monotonic_increasing:
        raise ValueError("X.Time must be monotonic nondecreasing for temporal CV")

    for effective_splits in range(n_splits, 1, -1):
        try:
            folds = list(TimeSeriesSplit(n_splits=effective_splits).split(X))
        except ValueError:
            continue
        if any(
            y.iloc[train_indices].nunique() < 2 or y.iloc[val_indices].nunique() < 2
            for train_indices, val_indices in folds
        ):
            continue

        scores: list[float] = []
        for train_indices, val_indices in folds:
            raw_train = X.iloc[train_indices]
            raw_val = X.iloc[val_indices]
            train_target = y.iloc[train_indices]
            val_target = y.iloc[val_indices]
            pipeline = FeaturePipeline().fit(raw_train)
            train_features = pipeline.transform(raw_train)
            val_features = pipeline.transform(raw_val)
            weight = (
                compute_scale_pos_weight(train_target)
                if imbalance_strategy == "scale-pos-weight"
                else None
            )
            estimator = build_estimator(weight)
            estimator.fit(train_features, train_target)
            scores.append(
                float(
                    average_precision_score(
                        val_target, estimator.predict_proba(val_features)[:, 1]
                    )
                )
            )
        return scores, effective_splits

    raise ValueError(
        "no valid temporal CV with at least two folds; each fold train and validation must contain both classes"
    )


def _run_optuna_study(
    objective: Callable[[optuna.Trial], float], n_trials: int, random_state: int
) -> optuna.Study:
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def _suggest_params(trial: optuna.Trial, space: dict[str, tuple[list, str]]) -> dict[str, Any]:
    return {
        name: trial.suggest_categorical(name, choices)
        for name, (choices, _) in space.items()
    }


class _TunedRandomForestFactory:
    def __init__(self, best_params: dict[str, Any], random_state: int) -> None:
        self._best_params = best_params
        self._random_state = random_state

    def create(self, scale_pos_weight: float | None = None) -> RandomForestClassifier:
        class_weight = None if scale_pos_weight is None else {0: 1.0, 1: scale_pos_weight}
        return RandomForestClassifier(
            **self._best_params,
            random_state=self._random_state,
            class_weight=class_weight,
        )


class _TunedXGBoostFactory:
    def __init__(self, best_params: dict[str, Any], random_state: int) -> None:
        self._best_params = best_params
        self._random_state = random_state

    def create(self, scale_pos_weight: float | None = None) -> Classifier:
        from xgboost import XGBClassifier

        extra = {} if scale_pos_weight is None else {"scale_pos_weight": scale_pos_weight}
        return XGBClassifier(
            **self._best_params,
            random_state=self._random_state,
            eval_metric="logloss",
            verbosity=0,
            **extra,
        )


class _TunedLightGbmFactory:
    def __init__(self, best_params: dict[str, Any], random_state: int) -> None:
        self._best_params = best_params
        self._random_state = random_state

    def create(self, scale_pos_weight: float | None = None) -> Classifier:
        from lightgbm import LGBMClassifier

        extra = {} if scale_pos_weight is None else {"scale_pos_weight": scale_pos_weight}
        return LGBMClassifier(
            **self._best_params,
            random_state=self._random_state,
            verbose=-1,
            **extra,
        )


@dataclass(frozen=True)
class TuningResult:
    best_params: dict[str, Any]
    best_score: float
    scoring: str
    best_factory: ModelFactory
    cv_splits: int
    random_state: int
    cv_strategy: str = "TimeSeriesSplit"


def _tune(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_iter: int,
    cv: int,
    scoring: str,
    random_state: int,
    imbalance_strategy: ImbalanceStrategy,
    parameter_space: dict[str, tuple[list, str]],
    build_candidate: Callable[[dict[str, Any], float | None], Classifier],
    build_factory: Callable[[dict[str, Any], int], ModelFactory],
) -> TuningResult:
    _validate_tuning_inputs(X, y, n_iter=n_iter, cv=cv, scoring=scoring)
    effective_cv = cv

    def objective(trial: optuna.Trial) -> float:
        nonlocal effective_cv
        params = _suggest_params(trial, parameter_space)
        scores, effective_cv = _temporal_cv_scores(
            X,
            y,
            n_splits=cv,
            imbalance_strategy=imbalance_strategy,
            build_estimator=lambda weight: build_candidate(params, weight),
        )
        return float(np.mean(scores))

    study = _run_optuna_study(objective, n_iter, random_state)
    return TuningResult(
        best_params=study.best_params,
        best_score=float(study.best_value),
        scoring=scoring,
        best_factory=build_factory(study.best_params, random_state),
        cv_splits=effective_cv,
        random_state=random_state,
    )


def tune_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_iter: int = 10,
    cv: int = 3,
    scoring: str = "average_precision",
    random_state: int = 42,
    imbalance_strategy: ImbalanceStrategy = "none",
) -> TuningResult:
    def build(params: dict[str, Any], weight: float | None) -> Classifier:
        class_weight = None if weight is None else {0: 1.0, 1: weight}
        return RandomForestClassifier(
            **params, random_state=random_state, class_weight=class_weight
        )

    return _tune(
        X,
        y,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
        imbalance_strategy=imbalance_strategy,
        parameter_space=_RF_PARAM_SPACE,
        build_candidate=build,
        build_factory=_TunedRandomForestFactory,
    )


def tune_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_iter: int = 10,
    cv: int = 3,
    scoring: str = "average_precision",
    random_state: int = 42,
    imbalance_strategy: ImbalanceStrategy = "none",
) -> TuningResult:
    def build(params: dict[str, Any], weight: float | None) -> Classifier:
        from xgboost import XGBClassifier

        extra = {} if weight is None else {"scale_pos_weight": weight}
        return XGBClassifier(
            **params,
            random_state=random_state,
            eval_metric="logloss",
            verbosity=0,
            **extra,
        )

    return _tune(
        X,
        y,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
        imbalance_strategy=imbalance_strategy,
        parameter_space=_XGB_PARAM_SPACE,
        build_candidate=build,
        build_factory=_TunedXGBoostFactory,
    )


def tune_lightgbm(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_iter: int = 10,
    cv: int = 3,
    scoring: str = "average_precision",
    random_state: int = 42,
    imbalance_strategy: ImbalanceStrategy = "none",
) -> TuningResult:
    def build(params: dict[str, Any], weight: float | None) -> Classifier:
        from lightgbm import LGBMClassifier

        extra = {} if weight is None else {"scale_pos_weight": weight}
        return LGBMClassifier(
            **params,
            random_state=random_state,
            verbose=-1,
            **extra,
        )

    return _tune(
        X,
        y,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
        imbalance_strategy=imbalance_strategy,
        parameter_space=_LGBM_PARAM_SPACE,
        build_candidate=build,
        build_factory=_TunedLightGbmFactory,
    )
