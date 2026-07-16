import json

import numpy as np
import pandas as pd
import pytest

from fraud_detection.cli import main


def _csv(tmp_path, n: int = 20):
    path = tmp_path / "creditcard.csv"
    pd.DataFrame(
        {
            "Time": np.arange(n, dtype=float),
            "Amount": np.resize([1.0, 100.0], n),
            "V1": np.resize([0.0, 4.0], n),
            "Class": np.resize([0, 1], n),
        }
    ).to_csv(path, index=False)
    return path


def test_cli_writes_structured_run_metadata_from_effective_argv(tmp_path, capsys):
    path = _csv(tmp_path)
    artifact_dir = tmp_path / "runs"
    argv = [
        "--data-path",
        str(path),
        "--batch-size",
        "20",
        "--test-size",
        "0.2",
        "--model",
        "logistic-regression",
        "--seed",
        "7",
        "--artifact-dir",
        str(artifact_dir),
    ]

    main(argv)

    output = capsys.readouterr().out
    run_dir = next(artifact_dir.iterdir())
    config = json.loads((run_dir / "config.json").read_text())
    assert f"artifacts saved to {run_dir}" in output
    assert config["command"] == ["fraud-detect-train", *argv]
    assert config["dataset"]["rows"] == 20
    assert config["run"]["seed"] == 7
    assert config["run"]["duration_s"] >= 0
    assert config["training"]["model_key"] == "logistic-regression"
    assert config["training"]["estimator_class"] == "LogisticRegression"
    assert config["split"]["audit"]["train"]["positives"] > 0
    assert config["split"]["audit"]["test"]["negatives"] > 0
    assert config["threshold"]["selected"] == pytest.approx(0.5)
    assert config["threshold"]["objective"] == "fixed"


def test_cli_validation_threshold_is_the_persisted_operational_threshold(tmp_path):
    path = _csv(tmp_path)
    artifact_dir = tmp_path / "runs"
    main(
        [
            "--data-path",
            str(path),
            "--batch-size",
            "20",
            "--test-size",
            "0.2",
            "--val-size",
            "0.2",
            "--threshold-objective",
            "target-recall",
            "--target-recall",
            "0.95",
            "--model",
            "logistic-regression",
            "--artifact-dir",
            str(artifact_dir),
        ]
    )

    config = json.loads((next(artifact_dir.iterdir()) / "config.json").read_text())
    assert 0 < config["threshold"]["selected"] < 1
    assert config["threshold"]["target_recall"] == pytest.approx(0.95)
    assert isinstance(config["threshold"]["fallback_used"], bool)
