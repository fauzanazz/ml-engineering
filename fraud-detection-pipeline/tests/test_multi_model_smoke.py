"""Integration smoke tests: all model factories train and produce predictions."""
import numpy as np
import pandas as pd
import pytest

from fraud_detection.models import (
    DecisionTreeFactory,
    LightGbmFactory,
    LogisticRegressionFactory,
    RandomForestFactory,
    XGBoostFactory,
)
from fraud_detection.training import train_one_batch


def _minimal_csv(tmp_path):
    data_path = tmp_path / "creditcard.csv"
    pd.DataFrame(
        {
            "Time": [1, 2, 3, 4, 5, 6],
            "V1": [0.0, 4.2, 0.1, 4.1, 0.2, 4.0],
            "V2": [0.0, 4.2, 0.1, 4.1, 0.2, 4.0],
            "Amount": [1.0, 102.0, 1.1, 101.0, 1.2, 100.0],
            "Class": [0, 1, 0, 1, 0, 1],
        }
    ).to_csv(data_path, index=False)
    return data_path


@pytest.mark.parametrize(
    "factory",
    [
        LightGbmFactory(),
        LogisticRegressionFactory(),
        DecisionTreeFactory(),
        RandomForestFactory(),
        XGBoostFactory(),
    ],
    ids=["lightgbm", "logistic-regression", "decision-tree", "random-forest", "xgboost"],
)
def test_all_model_factories_train_and_produce_predictions(tmp_path, factory):
    result = train_one_batch(
        data_path=_minimal_csv(tmp_path),
        model_factory=factory,
        batch_size=6,
        test_size=0.5,
    )

    assert len(result.predictions) == 3
    assert all(p in (0, 1) for p in result.predictions)
    assert 0.0 <= result.training_accuracy <= 1.0
    assert 0.0 <= result.test_accuracy <= 1.0


@pytest.mark.parametrize(
    "factory",
    [
        LightGbmFactory(),
        LogisticRegressionFactory(),
        DecisionTreeFactory(),
        RandomForestFactory(),
        XGBoostFactory(),
    ],
    ids=["lightgbm", "logistic-regression", "decision-tree", "random-forest", "xgboost"],
)
def test_all_model_factories_work_with_scale_pos_weight(tmp_path, factory):
    result = train_one_batch(
        data_path=_minimal_csv(tmp_path),
        model_factory=factory,
        batch_size=6,
        test_size=0.5,
        imbalance_strategy="scale-pos-weight",
    )

    assert len(result.predictions) == 3
    assert 0.0 <= result.training_accuracy <= 1.0
