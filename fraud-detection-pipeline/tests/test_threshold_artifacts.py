import json

import pytest

from fraud_detection.artifacts import write_artifacts
from fraud_detection.metrics import ClassificationMetrics
from fraud_detection.training import TrainingResult


@pytest.fixture()
def sample_result():
    return TrainingResult(
        predictions=[],
        training_accuracy=0.999,
        test_accuracy=0.998,
        metrics=ClassificationMetrics(precision=0.81, recall=1.0, f1=0.90, pr_auc=0.97),
    )


def test_config_includes_decision_threshold(tmp_path, sample_result):
    config = {
        "data_path": "data/creditcard.csv",
        "batch_size": 256,
        "test_size": 0.2,
        "imbalance_strategy": "none",
        "model_name": "LightGbmFactory",
        "decision_threshold": 0.3,
    }
    run_dir = tmp_path / "run-threshold"

    write_artifacts(run_dir, result=sample_result, config=config)

    data = json.loads((run_dir / "config.json").read_text())
    assert data["decision_threshold"] == pytest.approx(0.3)
