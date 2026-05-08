import numpy as np
import pytest

from fraud_detection.metrics import ClassificationMetrics, SklearnMetricsAdapter


@pytest.fixture
def adapter():
    return SklearnMetricsAdapter()


def test_perfect_predictions_give_all_ones(adapter):
    labels = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.9, 0.1, 0.9])

    metrics = adapter.compute(labels, predictions=np.array([0, 1, 0, 1]), scores=scores)

    assert metrics.precision == pytest.approx(1.0)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.f1 == pytest.approx(1.0)
    assert metrics.pr_auc == pytest.approx(1.0)


def test_all_wrong_predictions_give_zeros_not_errors(adapter):
    labels = np.array([1, 1, 0, 0])
    predictions = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.1, 0.9, 0.9])

    metrics = adapter.compute(labels, predictions=predictions, scores=scores)

    assert metrics.precision == pytest.approx(0.0)
    assert metrics.recall == pytest.approx(0.0)
    assert metrics.f1 == pytest.approx(0.0)


def test_pr_auc_uses_scores_not_predictions(adapter):
    labels = np.array([0, 0, 1, 1])
    predictions = np.array([0, 0, 0, 0])  # all wrong class predictions
    scores = np.array([0.1, 0.2, 0.8, 0.9])  # but good scores

    metrics = adapter.compute(labels, predictions=predictions, scores=scores)

    assert metrics.pr_auc > 0.5


def test_metrics_is_frozen_dataclass(adapter):
    labels = np.array([0, 1])
    predictions = np.array([0, 1])
    scores = np.array([0.1, 0.9])

    metrics = adapter.compute(labels, predictions=predictions, scores=scores)

    assert isinstance(metrics, ClassificationMetrics)
    with pytest.raises((AttributeError, TypeError)):
        metrics.precision = 0.5  # type: ignore[misc]


def test_compute_without_scores_raises(adapter):
    labels = np.array([0, 1])
    predictions = np.array([0, 1])

    with pytest.raises(TypeError):
        adapter.compute(labels, predictions=predictions)
