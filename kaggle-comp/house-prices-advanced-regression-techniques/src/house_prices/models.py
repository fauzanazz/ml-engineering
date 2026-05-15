from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import BayesianRidge, ElasticNet, Lasso
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor


@runtime_checkable
class Regressor(Protocol):
    def fit(self, X, y): ...
    def predict(self, X) -> np.ndarray: ...


class ModelFactory(Protocol):
    def create(self) -> Regressor: ...


class LightGbmFactory:
    def __init__(self, random_state: int = 42) -> None:
        self._random_state = random_state

    def create(self) -> Regressor:
        return LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            random_state=self._random_state,
            verbose=-1,
        )


class RandomForestFactory:
    def __init__(self, random_state: int = 42) -> None:
        self._random_state = random_state

    def create(self) -> Regressor:
        return RandomForestRegressor(
            n_estimators=300,
            random_state=self._random_state,
            n_jobs=-1,
        )


class XGBoostFactory:
    def __init__(self, random_state: int = 42) -> None:
        self._random_state = random_state

    def create(self) -> Regressor:
        return XGBRegressor(
            n_estimators=1661,
            learning_rate=0.03878060570513331,
            max_depth=3,
            min_child_weight=1.7740742531605194,
            subsample=0.6837885398447652,
            colsample_bytree=0.8040431594420846,
            reg_lambda=0.07772592475736047,
            reg_alpha=0.0004111444345783193,
            random_state=self._random_state,
            objective="reg:squarederror",
            n_jobs=-1,
        )

class WeightedLogEnsemble:
    def __init__(self, estimators: list[tuple[float, Regressor]]) -> None:
        total_weight = sum(weight for weight, _ in estimators)
        if total_weight <= 0:
            raise ValueError("ensemble weights must sum to a positive value")
        self._estimators = [(weight / total_weight, estimator) for weight, estimator in estimators]

    def fit(self, X, y):
        for _, estimator in self._estimators:
            estimator.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        predictions = [weight * estimator.predict(X) for weight, estimator in self._estimators]
        return np.sum(predictions, axis=0)


class ResidualCorrectedLogEnsemble:
    def __init__(self, base_model: Regressor, residual_model: Regressor, residual_weight: float) -> None:
        self._base_model = base_model
        self._residual_model = residual_model
        self._residual_weight = residual_weight

    def fit(self, X, y):
        self._base_model.fit(X, y)
        self._residual_model.fit(X, y - self._base_model.predict(X))
        return self

    def predict(self, X) -> np.ndarray:
        return self._base_model.predict(X) + self._residual_weight * self._residual_model.predict(X)


class SequentialGroupResidualRegressor:
    def __init__(self, corrections: list[tuple[str, float, float]]) -> None:
        self._corrections = corrections
        self._mappings: list[tuple[str, float, pd.Series]] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        residual = np.asarray(y, dtype=float).copy()
        self._mappings = []
        for column, smoothing, weight in self._corrections:
            grouped = pd.DataFrame({"key": X[column], "residual": residual}).groupby("key")["residual"].agg(["mean", "count"])
            mapping = grouped["mean"] * grouped["count"] / (grouped["count"] + smoothing)
            residual = residual - weight * X[column].map(mapping).fillna(0).to_numpy()
            self._mappings.append((column, weight, mapping))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        prediction = np.zeros(len(X), dtype=float)
        for column, weight, mapping in self._mappings:
            prediction += weight * X[column].map(mapping).fillna(0).to_numpy()
        return prediction

class WeightedEnsembleFactory:
    def __init__(self, random_state: int = 42) -> None:
        self._random_state = random_state

    def create(self) -> Regressor:
        base_model = WeightedLogEnsemble(
            [
                (0.2975, XGBoostFactory(random_state=self._random_state).create()),
                (0.170, ElasticNet(alpha=0.001, l1_ratio=0.2, max_iter=20000)),
                (0.085, Lasso(alpha=0.0005, max_iter=20000)),
                (
                    0.2975,
                    GradientBoostingRegressor(
                        n_estimators=1500,
                        learning_rate=0.02,
                        max_depth=3,
                        subsample=0.75,
                        random_state=self._random_state,
                    ),
                ),
                (0.150, BayesianRidge()),
            ]
        )
        residual_model = RandomForestRegressor(
            n_estimators=800,
            min_samples_leaf=15,
            max_features=0.4,
            random_state=self._random_state,
            n_jobs=-1,
        )
        random_forest_corrected = ResidualCorrectedLogEnsemble(base_model, residual_model, residual_weight=2.75)
        neighbor_residual_model = KNeighborsRegressor(n_neighbors=15)
        neighbor_corrected = ResidualCorrectedLogEnsemble(random_forest_corrected, neighbor_residual_model, residual_weight=-1.25)
        group_residual_model = SequentialGroupResidualRegressor(
            [
                ("Fence_GdWo", 1, 4.0),
                ("Condition1_RRAe", 1, 4.0),
                ("Condition2_Norm", 1, -3.0),
                ("MSSubClass", 100, 2.0),
                ("Condition2_Feedr", 1, -3.0),
                ("GarageType_Detchd", 1, -3.0),
                ("Electrical_SBrkr", 1, -2.0),
                ("FireplaceQuScore", 100, 1.0),
                ("BsmtFinType2_Unf", 5, -3.0),
                ("SaleCondition_Normal", 100, 1.0),
                ("Exterior1st_Wd Sdng", 1, -3.0),
                ("SaleCondition_Abnorml", 1, -2.0),
            ]
        )
        return ResidualCorrectedLogEnsemble(neighbor_corrected, group_residual_model, residual_weight=1.0)


def make_model_factory(name: str, random_state: int = 42) -> ModelFactory:
    factories = {
        "ensemble": WeightedEnsembleFactory,
        "lightgbm": LightGbmFactory,
        "random-forest": RandomForestFactory,
        "xgboost": XGBoostFactory,
    }
    if name not in factories:
        valid_names = ", ".join(sorted(factories))
        raise ValueError(f"Unknown model '{name}'. Valid models: {valid_names}")
    return factories[name](random_state=random_state)
