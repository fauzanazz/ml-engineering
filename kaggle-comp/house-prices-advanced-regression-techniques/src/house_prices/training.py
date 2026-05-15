from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from house_prices.data import load_train_validation_split
from house_prices.features import FeaturePipeline
from house_prices.metrics import RegressionMetrics, SklearnRegressionMetrics
from house_prices.models import ModelFactory


@dataclass(frozen=True)
class TrainingResult:
    predictions: list[float]
    metrics: RegressionMetrics
    model: object
    feature_pipeline: FeaturePipeline
    validation_labels: list[float]
    split_strategy: str
    target_transform: str


def train_model(
    data_path: Path | str,
    model_factory: ModelFactory,
    *,
    target_column: str = "SalePrice",
    val_size: float = 0.2,
    random_state: int = 42,
) -> TrainingResult:
    train_features, val_features, train_target, val_target, split_strategy = load_train_validation_split(
        data_path,
        target_column=target_column,
        val_size=val_size,
        random_state=random_state,
    )

    feature_pipeline = FeaturePipeline().fit(train_features, train_target)
    transformed_train = feature_pipeline.transform(train_features)
    transformed_val = feature_pipeline.transform(val_features)

    model = model_factory.create()
    model.fit(transformed_train, np.log(train_target))

    log_predictions = model.predict(transformed_val)
    predictions = np.exp(log_predictions)
    metrics = SklearnRegressionMetrics().compute(val_target.to_numpy(), predictions)

    return TrainingResult(
        predictions=np.asarray(predictions, dtype=float).tolist(),
        metrics=metrics,
        model=model,
        feature_pipeline=feature_pipeline,
        validation_labels=val_target.astype(float).tolist(),
        split_strategy=split_strategy,
        target_transform="log",
    )


def predict_submission(
    model,
    feature_pipeline: FeaturePipeline,
    test_features: pd.DataFrame,
    *,
    id_column: str = "Id",
) -> pd.DataFrame:
    if id_column not in test_features.columns:
        raise ValueError(f"Missing id column: {id_column}")

    transformed_test = feature_pipeline.transform(test_features)
    predictions = np.maximum(np.exp(model.predict(transformed_test)), 0)
    return pd.DataFrame({id_column: test_features[id_column], "SalePrice": predictions})
