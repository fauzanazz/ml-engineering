import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from fraud_detection.models import Classifier, ModelFactory

# Suppress optuna's per-trial INFO logs — keep warnings/errors visible.
optuna.logging.set_verbosity(logging.WARNING)


def _validate_tuning_inputs(
    X: pd.DataFrame, y: pd.Series, n_iter: int, scoring: str, cv: int | None = None
) -> int:
    """Validate inputs and return safe cv fold count.

    Raises ValueError on unrecoverable conditions so callers can surface clean errors.
    """
    if n_iter < 1:
        raise ValueError(f"n_iter must be >= 1, got {n_iter}")

    counts = y.value_counts()
    if len(counts) < 2:
        raise ValueError(
            f"y must contain at least 2 classes for {scoring}; found only class(es): {list(counts.index)}"
        )
    min_count = int(counts.min())
    if min_count < 2:
        raise ValueError(
            f"Each class needs at least 2 samples for CV; minority class has {min_count} sample(s)"
        )

    safe_cv = max(2, min(3, min_count))

    if cv is not None:
        if cv < 2:
            raise ValueError(f"cv must be >= 2, got {cv}")
        if cv > min_count:
            raise ValueError(
                f"cv={cv} exceeds minority class count ({min_count}); "
                f"each fold needs at least one minority sample"
            )

    return safe_cv


def _make_cv(n_splits: int, random_state: int) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def _run_optuna_study(
    objective_fn: Any,
    n_trials: int,
    random_state: int,
) -> optuna.Study:
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=False)
    return study


_RF_PARAM_SPACE = {
    "n_estimators": ([50, 100, 200], "categorical"),
    "max_depth": ([None, 5, 10, 20], "categorical"),
    "min_samples_split": ([2, 5, 10], "categorical"),
    "min_samples_leaf": ([1, 2, 4], "categorical"),
    "max_features": (["sqrt", "log2", None], "categorical"),
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
    "max_depth": ([3, 5, 7, -1], "categorical"),
    "learning_rate": ([0.01, 0.05, 0.1, 0.2, 0.3], "categorical"),
    "num_leaves": ([15, 31, 63, 127], "categorical"),
    "subsample": ([0.6, 0.8, 1.0], "categorical"),
}


def _suggest_params(trial: optuna.Trial, space: dict[str, tuple]) -> dict[str, Any]:
    return {name: trial.suggest_categorical(name, choices) for name, (choices, _) in space.items()}


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

        extra: dict[str, float] = {} if scale_pos_weight is None else {"scale_pos_weight": scale_pos_weight}
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

        extra: dict[str, float] = {} if scale_pos_weight is None else {"scale_pos_weight": scale_pos_weight}
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


def tune_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_iter: int = 10,
    cv: int | None = None,
    scoring: str = "roc_auc",
    random_state: int = 42,
    n_jobs: int = 1,
) -> TuningResult:
    safe_cv = _validate_tuning_inputs(X, y, n_iter, scoring, cv)
    effective_cv = cv if cv is not None else safe_cv
    cv_splitter = _make_cv(effective_cv, random_state)

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, _RF_PARAM_SPACE)
        estimator = RandomForestClassifier(**params, random_state=random_state)
        scores = cross_val_score(estimator, X, y, cv=cv_splitter, scoring=scoring, n_jobs=n_jobs)
        return float(np.mean(scores))

    study = _run_optuna_study(objective, n_iter, random_state)
    best_params = study.best_params
    return TuningResult(
        best_params=best_params,
        best_score=study.best_value,
        scoring=scoring,
        best_factory=_TunedRandomForestFactory(best_params, random_state),
    )


def tune_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_iter: int = 10,
    cv: int | None = None,
    scoring: str = "roc_auc",
    random_state: int = 42,
    n_jobs: int = 1,
) -> TuningResult:
    from xgboost import XGBClassifier

    safe_cv = _validate_tuning_inputs(X, y, n_iter, scoring, cv)
    effective_cv = cv if cv is not None else safe_cv
    cv_splitter = _make_cv(effective_cv, random_state)

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, _XGB_PARAM_SPACE)
        estimator = XGBClassifier(
            **params,
            random_state=random_state,
            eval_metric="logloss",
            verbosity=0,
        )
        scores = cross_val_score(estimator, X, y, cv=cv_splitter, scoring=scoring, n_jobs=n_jobs)
        return float(np.mean(scores))

    study = _run_optuna_study(objective, n_iter, random_state)
    best_params = study.best_params
    return TuningResult(
        best_params=best_params,
        best_score=study.best_value,
        scoring=scoring,
        best_factory=_TunedXGBoostFactory(best_params, random_state),
    )


def tune_lightgbm(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_iter: int = 10,
    cv: int | None = None,
    scoring: str = "roc_auc",
    random_state: int = 42,
    n_jobs: int = 1,
) -> TuningResult:
    from lightgbm import LGBMClassifier

    safe_cv = _validate_tuning_inputs(X, y, n_iter, scoring, cv)
    effective_cv = cv if cv is not None else safe_cv
    cv_splitter = _make_cv(effective_cv, random_state)

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, _LGBM_PARAM_SPACE)
        estimator = LGBMClassifier(**params, random_state=random_state, verbose=-1)
        scores = cross_val_score(estimator, X, y, cv=cv_splitter, scoring=scoring, n_jobs=n_jobs)
        return float(np.mean(scores))

    study = _run_optuna_study(objective, n_iter, random_state)
    best_params = study.best_params
    return TuningResult(
        best_params=best_params,
        best_score=study.best_value,
        scoring=scoring,
        best_factory=_TunedLightGbmFactory(best_params, random_state),
    )
