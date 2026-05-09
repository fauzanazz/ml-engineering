from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from fraud_detection.models import Classifier, ModelFactory


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


_RF_PARAM_DIST: dict[str, Any] = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
}

_XGB_PARAM_DIST: dict[str, Any] = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}

_LGBM_PARAM_DIST: dict[str, Any] = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7, -1],
    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
    "num_leaves": [15, 31, 63, 127],
    "subsample": [0.6, 0.8, 1.0],
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
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=random_state),
        param_distributions=_RF_PARAM_DIST,
        n_iter=n_iter,
        scoring=scoring,
        cv=_make_cv(effective_cv, random_state),
        random_state=random_state,
        n_jobs=n_jobs,
    )
    search.fit(X, y)
    return TuningResult(
        best_params=search.best_params_,
        best_score=float(search.best_score_),
        scoring=scoring,
        best_factory=_TunedRandomForestFactory(search.best_params_, random_state),
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
    search = RandomizedSearchCV(
        XGBClassifier(
            random_state=random_state,
            eval_metric="logloss",
            verbosity=0,
        ),
        param_distributions=_XGB_PARAM_DIST,
        n_iter=n_iter,
        scoring=scoring,
        cv=_make_cv(effective_cv, random_state),
        random_state=random_state,
        n_jobs=n_jobs,
    )
    search.fit(X, y)
    return TuningResult(
        best_params=search.best_params_,
        best_score=float(search.best_score_),
        scoring=scoring,
        best_factory=_TunedXGBoostFactory(search.best_params_, random_state),
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
    search = RandomizedSearchCV(
        LGBMClassifier(random_state=random_state, verbose=-1),
        param_distributions=_LGBM_PARAM_DIST,
        n_iter=n_iter,
        scoring=scoring,
        cv=_make_cv(effective_cv, random_state),
        random_state=random_state,
        n_jobs=n_jobs,
    )
    search.fit(X, y)
    return TuningResult(
        best_params=search.best_params_,
        best_score=float(search.best_score_),
        scoring=scoring,
        best_factory=_TunedLightGbmFactory(search.best_params_, random_state),
    )
