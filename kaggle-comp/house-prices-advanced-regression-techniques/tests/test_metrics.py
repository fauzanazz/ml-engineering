import numpy as np

from house_prices.metrics import SklearnRegressionMetrics


def test_regression_metrics_are_finite_for_positive_predictions():
    labels = np.array([100_000, 150_000, 200_000])
    predictions = np.array([110_000, 145_000, 190_000])

    metrics = SklearnRegressionMetrics().compute(labels, predictions)

    assert metrics.mae > 0
    assert metrics.rmse > 0
    assert metrics.log_rmse > 0
    assert metrics.r2 > 0
