import json
from pathlib import Path

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
        metrics=ClassificationMetrics(
            precision=0.81,
            recall=1.0,
            f1=0.90,
            pr_auc=0.97,
        ),
    )


@pytest.fixture()
def sample_config():
    return {
        "data_path": "data/creditcard.csv",
        "batch_size": 10000,
        "test_size": 0.2,
        "imbalance_strategy": "none",
        "model_name": "LightGbmFactory",
    }


def test_write_artifacts_creates_metrics_json(tmp_path, sample_result, sample_config):
    run_dir = tmp_path / "run-001"

    write_artifacts(run_dir, result=sample_result, config=sample_config)

    metrics_file = run_dir / "metrics.json"
    assert metrics_file.exists()
    data = json.loads(metrics_file.read_text())
    assert data["training_accuracy"] == pytest.approx(0.999)
    assert data["test_accuracy"] == pytest.approx(0.998)
    assert data["precision"] == pytest.approx(0.81)
    assert data["recall"] == pytest.approx(1.0)
    assert data["f1"] == pytest.approx(0.90)
    assert data["pr_auc"] == pytest.approx(0.97)


def test_write_artifacts_creates_config_json(tmp_path, sample_result, sample_config):
    run_dir = tmp_path / "run-001"

    write_artifacts(run_dir, result=sample_result, config=sample_config)

    config_file = run_dir / "config.json"
    assert config_file.exists()
    data = json.loads(config_file.read_text())
    assert data["data_path"] == "data/creditcard.csv"
    assert data["batch_size"] == 10000
    assert data["imbalance_strategy"] == "none"
    assert data["model_name"] == "LightGbmFactory"


def test_write_artifacts_creates_run_dir_if_missing(tmp_path, sample_result, sample_config):
    run_dir = tmp_path / "nested" / "run-001"
    assert not run_dir.exists()

    write_artifacts(run_dir, result=sample_result, config=sample_config)

    assert run_dir.exists()


def test_write_artifacts_saves_model_txt_when_model_provided(
    tmp_path, sample_result, sample_config
):
    from lightgbm import LGBMClassifier
    import numpy as np

    model = LGBMClassifier(n_estimators=5, num_leaves=4, verbose=-1, random_state=42)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 4))
    y = np.array([0] * 40 + [1] * 10)
    model.fit(X, y)

    run_dir = tmp_path / "run-model"
    write_artifacts(run_dir, result=sample_result, config=sample_config, model=model)

    assert (run_dir / "model.txt").exists()
    assert (run_dir / "model.txt").stat().st_size > 0


def test_write_artifacts_skips_model_txt_when_no_model(tmp_path, sample_result, sample_config):
    run_dir = tmp_path / "run-no-model"

    write_artifacts(run_dir, result=sample_result, config=sample_config, model=None)

    assert not (run_dir / "model.txt").exists()
