import json
from pathlib import Path

import pytest

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


def test_parse_args_default_artifact_dir_is_artifacts_runs():
    from fraud_detection.cli import parse_args

    args = parse_args(["--data-path", "data/creditcard.csv"])

    assert args.artifact_dir == Path("artifacts/runs")


def test_parse_args_accepts_custom_artifact_dir():
    from fraud_detection.cli import parse_args

    args = parse_args(["--data-path", "data/creditcard.csv", "--artifact-dir", "/tmp/myexp"])

    assert args.artifact_dir == Path("/tmp/myexp")


def test_parse_args_accepts_no_artifact_dir_flag():
    from fraud_detection.cli import parse_args

    args = parse_args(["--data-path", "data/creditcard.csv", "--no-artifacts"])

    assert args.no_artifacts is True


def test_make_run_id_is_deterministic_when_provided():
    from fraud_detection.artifacts import make_run_dir

    base = Path("/tmp/runs")
    run_dir = make_run_dir(base, run_id="test-run-42")

    assert run_dir == base / "test-run-42"


def test_make_run_id_uses_timestamp_when_not_provided():
    from fraud_detection.artifacts import make_run_dir

    base = Path("/tmp/runs")
    run_dir = make_run_dir(base)

    # timestamped: starts with base path, name has digits
    assert run_dir.parent == base
    assert any(ch.isdigit() for ch in run_dir.name)


def test_two_generated_run_dirs_are_unique():
    from fraud_detection.artifacts import make_run_dir

    base = Path("/tmp/runs")
    dir1 = make_run_dir(base)
    dir2 = make_run_dir(base)

    assert dir1 != dir2


def test_mkdir_raises_on_existing_run_dir(tmp_path):
    from fraud_detection.artifacts import make_run_dir

    base = tmp_path
    run_dir = make_run_dir(base, run_id="fixed-run")
    run_dir.mkdir(parents=True)

    with pytest.raises(FileExistsError):
        run_dir.mkdir(parents=True, exist_ok=False)


def test_write_artifacts_raises_on_existing_metrics_json(tmp_path, sample_result, sample_config):
    from fraud_detection.artifacts import write_artifacts

    run_dir = tmp_path / "run-collision"
    write_artifacts(run_dir, result=sample_result, config=sample_config)

    with pytest.raises(FileExistsError):
        write_artifacts(run_dir, result=sample_result, config=sample_config)


def test_write_artifacts_raises_on_existing_config_json(tmp_path, sample_result, sample_config):
    from fraud_detection.artifacts import write_artifacts

    run_dir = tmp_path / "run-collision-cfg"
    write_artifacts(run_dir, result=sample_result, config=sample_config)

    with pytest.raises(FileExistsError):
        write_artifacts(run_dir, result=sample_result, config=sample_config)


def test_write_artifacts_raises_on_existing_model_txt(tmp_path, sample_result, sample_config):
    from lightgbm import LGBMClassifier
    import numpy as np
    from fraud_detection.artifacts import write_artifacts

    model = LGBMClassifier(n_estimators=5, num_leaves=4, verbose=-1, random_state=42)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 4))
    y = np.array([0] * 40 + [1] * 10)
    model.fit(X, y)

    run_dir = tmp_path / "run-model-collision"
    write_artifacts(run_dir, result=sample_result, config=sample_config, model=model)

    with pytest.raises(FileExistsError):
        write_artifacts(run_dir, result=sample_result, config=sample_config, model=model)


def test_write_artifacts_returns_run_dir(tmp_path, sample_result, sample_config):
    from fraud_detection.artifacts import write_artifacts

    run_dir = tmp_path / "run-return"
    returned = write_artifacts(run_dir, result=sample_result, config=sample_config)

    assert returned == run_dir
