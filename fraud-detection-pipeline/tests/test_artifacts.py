import hashlib
import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from fraud_detection.artifacts import (
    collect_dataset_metadata,
    collect_runtime_metadata,
    write_artifacts,
)
from fraud_detection.models import LogisticRegressionFactory
from fraud_detection.training import train_one_batch


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


def _result(path):
    return train_one_batch(
        path,
        LogisticRegressionFactory(random_state=7),
        batch_size=20,
        test_size=0.2,
        val_size=0.2,
        threshold_objective="target-recall",
        target_recall=0.9,
        imbalance_strategy="scale-pos-weight",
    )


def _config(path, result):
    return {
        "schema_version": 1,
        "command": ["fraud-detect-train", "--data-path", str(path)],
        "dataset": collect_dataset_metadata(path, target_column="Class"),
        "run": {"started_at_utc": "2026-01-01T00:00:00+00:00", "duration_s": 1.2, "seed": 7},
        "runtime": collect_runtime_metadata(),
        "training": {
            "model_key": "logistic-regression",
            "estimator_class": type(result.model).__name__,
            "model_params": result.model.get_params(deep=False),
            "batch_size": 20,
            "imbalance_strategy": "scale-pos-weight",
        },
        "split": {"val_size": 0.2, "test_size": 0.2, "audit": asdict(result.split_audit)},
        "tuning": {"enabled": False},
        "threshold": {
            "objective": "target-recall",
            "target_recall": 0.9,
            "selected": result.effective_threshold,
            "target_met": result.threshold_target_met,
            "fallback_used": result.threshold_fallback_used,
        },
    }


def test_write_artifacts_emits_bundle_evaluation_and_versioned_json(tmp_path):
    path = _csv(tmp_path)
    result = _result(path)
    run_dir = tmp_path / "run"
    write_artifacts(run_dir, result=result, config=_config(path, result))

    assert {item.name for item in run_dir.iterdir()} == {
        "bundle.joblib",
        "evaluation.npz",
        "config.json",
        "metrics.json",
    }
    config = json.loads((run_dir / "config.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert set(config) == {
        "schema_version",
        "command",
        "dataset",
        "run",
        "runtime",
        "training",
        "split",
        "tuning",
        "threshold",
        "artifacts",
    }
    assert config["schema_version"] == metrics["schema_version"] == 1
    assert config["threshold"]["selected"] == pytest.approx(result.effective_threshold)
    assert config["split"]["audit"] == asdict(result.split_audit)
    assert metrics["test"] == asdict(result.metrics)
    assert metrics["validation"] == asdict(result.val_metrics)
    assert metrics["latency"]["warmup_calls"] == 1
    assert metrics["latency"]["single_row_repeats"] == 5
    assert metrics["latency"]["statistic"] == "median"


def test_artifact_manifests_match_written_bytes(tmp_path):
    path = _csv(tmp_path)
    result = _result(path)
    run_dir = tmp_path / "run"
    write_artifacts(run_dir, result=result, config=_config(path, result))
    config = json.loads((run_dir / "config.json").read_text())

    for name in ("bundle.joblib", "evaluation.npz"):
        expected = hashlib.sha256((run_dir / name).read_bytes()).hexdigest()
        assert config["artifacts"][name]["sha256"] == expected
    assert config["artifacts"]["bundle.joblib"]["format"] == "joblib"
    assert config["artifacts"]["bundle.joblib"]["trusted_load_required"] is True
    with np.load(run_dir / "evaluation.npz") as evaluation:
        assert set(evaluation.files) == {"test_labels", "test_scores"}


def test_collect_dataset_metadata_records_fingerprint_and_shape(tmp_path):
    path = _csv(tmp_path)
    metadata = collect_dataset_metadata(path, target_column="Class")

    assert metadata["rows"] == 20
    assert metadata["columns"] == ["Time", "Amount", "V1", "Class"]
    assert metadata["bytes"] == path.stat().st_size
    assert metadata["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_collect_runtime_metadata_has_required_versions():
    runtime = collect_runtime_metadata()
    assert set(runtime) == {
        "python",
        "platform",
        "joblib",
        "lightgbm",
        "numpy",
        "optuna",
        "pandas",
        "scikit-learn",
        "xgboost",
    }


def test_write_artifacts_rejects_incomplete_config(tmp_path):
    path = _csv(tmp_path)
    with pytest.raises(ValueError, match="top-level contract"):
        write_artifacts(tmp_path / "run", result=_result(path), config={})
