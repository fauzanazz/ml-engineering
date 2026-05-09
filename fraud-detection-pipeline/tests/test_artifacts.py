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
            roc_auc=0.95,
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
    assert data["roc_auc"] == pytest.approx(0.95)


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


def test_write_artifacts_saves_model_joblib_for_sklearn_model(
    tmp_path, sample_result, sample_config
):
    from sklearn.linear_model import LogisticRegression
    import numpy as np

    model = LogisticRegression(max_iter=1000, random_state=42)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 4))
    y = np.array([0] * 40 + [1] * 10)
    model.fit(X, y)

    run_dir = tmp_path / "run-sklearn"
    write_artifacts(run_dir, result=sample_result, config=sample_config, model=model)

    assert (run_dir / "model.joblib").exists()
    assert (run_dir / "model.joblib").stat().st_size > 0
    assert not (run_dir / "model.txt").exists()


def test_write_artifacts_joblib_adds_warning_to_metrics(
    tmp_path, sample_result, sample_config
):
    from sklearn.linear_model import LogisticRegression
    import numpy as np

    model = LogisticRegression(max_iter=1000, random_state=42)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 4))
    y = np.array([0] * 40 + [1] * 10)
    model.fit(X, y)

    run_dir = tmp_path / "run-joblib-warn"
    write_artifacts(run_dir, result=sample_result, config=sample_config, model=model)

    data = json.loads((run_dir / "metrics.json").read_text())
    assert "model_artifact_warning" in data
    assert "trusted" in data["model_artifact_warning"]


def test_write_artifacts_lgbm_no_warning_in_metrics(
    tmp_path, sample_result, sample_config
):
    from lightgbm import LGBMClassifier
    import numpy as np

    model = LGBMClassifier(n_estimators=5, num_leaves=4, verbose=-1, random_state=42)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 4))
    y = np.array([0] * 40 + [1] * 10)
    model.fit(X, y)

    run_dir = tmp_path / "run-lgbm-no-warn"
    write_artifacts(run_dir, result=sample_result, config=sample_config, model=model)

    data = json.loads((run_dir / "metrics.json").read_text())
    assert "model_artifact_warning" not in data


def test_write_artifacts_model_artifact_added_to_config_for_lgbm(
    tmp_path, sample_result, sample_config
):
    from lightgbm import LGBMClassifier
    import numpy as np

    model = LGBMClassifier(n_estimators=5, num_leaves=4, verbose=-1, random_state=42)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 4))
    y = np.array([0] * 40 + [1] * 10)
    model.fit(X, y)

    run_dir = tmp_path / "run-lgbm-artifact-cfg"
    write_artifacts(run_dir, result=sample_result, config=sample_config, model=model)

    data = json.loads((run_dir / "config.json").read_text())
    assert data["model_artifact"] == "model.txt"


def test_write_artifacts_model_artifact_added_to_config_for_joblib(
    tmp_path, sample_result, sample_config
):
    from sklearn.linear_model import LogisticRegression
    import numpy as np

    model = LogisticRegression(max_iter=1000, random_state=42)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 4))
    y = np.array([0] * 40 + [1] * 10)
    model.fit(X, y)

    run_dir = tmp_path / "run-joblib-artifact-cfg"
    write_artifacts(run_dir, result=sample_result, config=sample_config, model=model)

    data = json.loads((run_dir / "config.json").read_text())
    assert data["model_artifact"] == "model.joblib"
    assert "model_artifact_warning" in data
    assert "trusted" in data["model_artifact_warning"]


def test_write_artifacts_no_model_artifact_in_config_when_no_model(
    tmp_path, sample_result, sample_config
):
    run_dir = tmp_path / "run-no-artifact-cfg"
    write_artifacts(run_dir, result=sample_result, config=sample_config, model=None)

    data = json.loads((run_dir / "config.json").read_text())
    assert "model_artifact" not in data


def test_write_artifacts_skips_model_txt_when_no_model(tmp_path, sample_result, sample_config):
    run_dir = tmp_path / "run-no-model"

    write_artifacts(run_dir, result=sample_result, config=sample_config, model=None)

    assert not (run_dir / "model.txt").exists()


def test_write_artifacts_includes_val_metrics_when_present(tmp_path, sample_config):
    result = TrainingResult(
        predictions=[],
        training_accuracy=0.999,
        test_accuracy=0.998,
        metrics=ClassificationMetrics(precision=0.81, recall=1.0, f1=0.90, pr_auc=0.97, roc_auc=0.95),
        val_threshold=0.3,
        val_metrics=ClassificationMetrics(precision=0.75, recall=0.92, f1=0.82, pr_auc=0.88, roc_auc=0.91),
        threshold_objective="f1",
    )
    run_dir = tmp_path / "run-val"

    write_artifacts(run_dir, result=result, config=sample_config)

    data = json.loads((run_dir / "metrics.json").read_text())
    assert data["val_threshold"] == pytest.approx(0.3)
    assert data["val_precision"] == pytest.approx(0.75)
    assert data["val_recall"] == pytest.approx(0.92)
    assert data["val_f1"] == pytest.approx(0.82)
    assert data["val_pr_auc"] == pytest.approx(0.88)
    assert data["val_roc_auc"] == pytest.approx(0.91)


def test_write_artifacts_includes_latency_fields_in_metrics(tmp_path, sample_config):
    result = TrainingResult(
        predictions=[],
        training_accuracy=0.999,
        test_accuracy=0.998,
        metrics=ClassificationMetrics(precision=0.81, recall=1.0, f1=0.90, pr_auc=0.97, roc_auc=0.95),
        predict_proba_latency_s=0.0042,
        predict_proba_latency_per_row_s=0.0014,
    )
    run_dir = tmp_path / "run-latency"

    write_artifacts(run_dir, result=result, config=sample_config)

    data = json.loads((run_dir / "metrics.json").read_text())
    assert data["predict_proba_latency_s"] == pytest.approx(0.0042)
    assert data["predict_proba_latency_per_row_s"] == pytest.approx(0.0014)
    # backward-compat alias
    assert data["inference_latency_s"] == pytest.approx(0.0042)


def test_write_artifacts_includes_single_row_latency_in_metrics(tmp_path, sample_config):
    result = TrainingResult(
        predictions=[],
        training_accuracy=0.999,
        test_accuracy=0.998,
        metrics=ClassificationMetrics(precision=0.81, recall=1.0, f1=0.90, pr_auc=0.97, roc_auc=0.95),
        predict_proba_latency_s=0.0042,
        predict_proba_latency_per_row_s=0.0014,
        single_row_latency_s=0.00035,
    )
    run_dir = tmp_path / "run-single-row-latency"

    write_artifacts(run_dir, result=result, config=sample_config)

    data = json.loads((run_dir / "metrics.json").read_text())
    assert data["single_row_latency_s"] == pytest.approx(0.00035)


def test_write_artifacts_omits_single_row_latency_when_none(tmp_path, sample_result, sample_config):
    run_dir = tmp_path / "run-no-single-row"

    write_artifacts(run_dir, result=sample_result, config=sample_config)

    data = json.loads((run_dir / "metrics.json").read_text())
    assert "single_row_latency_s" not in data


def test_write_artifacts_omits_latency_fields_when_none(tmp_path, sample_result, sample_config):
    run_dir = tmp_path / "run-no-latency"

    write_artifacts(run_dir, result=sample_result, config=sample_config)

    data = json.loads((run_dir / "metrics.json").read_text())
    assert "predict_proba_latency_s" not in data
    assert "predict_proba_latency_per_row_s" not in data
    assert "inference_latency_s" not in data


def test_write_artifacts_omits_val_fields_when_absent(tmp_path, sample_result, sample_config):
    run_dir = tmp_path / "run-no-val"

    write_artifacts(run_dir, result=sample_result, config=sample_config)

    data = json.loads((run_dir / "metrics.json").read_text())
    assert "val_threshold" not in data
    assert "val_precision" not in data


def test_write_artifacts_config_includes_val_args_when_present(tmp_path, sample_result):
    config = {
        "data_path": "data/creditcard.csv",
        "batch_size": 10000,
        "test_size": 0.2,
        "val_size": 0.2,
        "threshold_objective": "target-recall",
        "target_recall": 0.95,
        "imbalance_strategy": "none",
        "model_name": "LightGbmFactory",
    }
    run_dir = tmp_path / "run-val-cfg"

    write_artifacts(run_dir, result=sample_result, config=config)

    data = json.loads((run_dir / "config.json").read_text())
    assert data["val_size"] == pytest.approx(0.2)
    assert data["threshold_objective"] == "target-recall"
    assert data["target_recall"] == pytest.approx(0.95)
