from collections.abc import Callable
from typing import Protocol, runtime_checkable

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


@runtime_checkable
class Classifier(Protocol):
    def fit(self, X, y): ...
    def predict(self, X) -> np.ndarray: ...
    def predict_proba(self, X) -> np.ndarray: ...


class ModelFactory(Protocol):
    def create(self, scale_pos_weight: float | None = None) -> Classifier:
        ...


def _class_weight(scale_pos_weight: float | None) -> dict | None:
    if scale_pos_weight is None:
        return None
    return {0: 1.0, 1: scale_pos_weight}


class LightGbmFactory:
    def __init__(self, scale_pos_weight: float | None = None) -> None:
        self._scale_pos_weight = scale_pos_weight

    def create(self, scale_pos_weight: float | None = None) -> LGBMClassifier:
        resolved = scale_pos_weight if scale_pos_weight is not None else self._scale_pos_weight
        extra = {} if resolved is None else {"scale_pos_weight": resolved}
        return LGBMClassifier(
            n_estimators=20,
            num_leaves=4,
            min_child_samples=1,
            random_state=42,
            verbose=-1,
            **extra,
        )


class LogisticRegressionFactory:
    def create(self, scale_pos_weight: float | None = None) -> LogisticRegression:
        return LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight=_class_weight(scale_pos_weight),
        )


class DecisionTreeFactory:
    def create(self, scale_pos_weight: float | None = None) -> DecisionTreeClassifier:
        return DecisionTreeClassifier(
            max_depth=2,
            random_state=42,
            class_weight=_class_weight(scale_pos_weight),
        )


class RandomForestFactory:
    def create(self, scale_pos_weight: float | None = None) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=20,
            random_state=42,
            class_weight=_class_weight(scale_pos_weight),
        )


class XGBoostFactory:
    def create(self, scale_pos_weight: float | None = None) -> XGBClassifier:
        extra: dict = {} if scale_pos_weight is None else {"scale_pos_weight": scale_pos_weight}
        return XGBClassifier(
            n_estimators=20,
            max_depth=4,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
            **extra,
        )


FACTORY_MAP: dict[str, Callable[[], ModelFactory]] = {
    "lightgbm": LightGbmFactory,
    "logistic-regression": LogisticRegressionFactory,
    "decision-tree": DecisionTreeFactory,
    "random-forest": RandomForestFactory,
    "xgboost": XGBoostFactory,
}
