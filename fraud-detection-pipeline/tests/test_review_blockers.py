"""Tests for review blocker fixes."""

import json

import numpy as np
import pandas as pd
import pytest

from fraud_detection.cli import parse_args
from fraud_detection.features import FeaturePipeline
from fraud_detection.artifacts import write_artifacts
from fraud_detection.metrics import ClassificationMetrics
from fraud_detection.training import SplitCounts, TrainingResult


# ---------------------------------------------------------------------------
# CLI: --target-recall range validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["-0.1", "1.1", "-1", "2"])
def test_parse_args_rejects_target_recall_out_of_range(bad):
    with pytest.raises(SystemExit):
        parse_args([
            "--data-path", "data/creditcard.csv",
            "--threshold-objective", "target-recall",
            "--val-size", "0.2",
            "--target-recall", bad,
        ])


@pytest.mark.parametrize("good", ["0", "0.0", "0.5", "0.95", "1", "1.0"])
def test_parse_args_accepts_valid_target_recall(good):
    args = parse_args([
        "--data-path", "data/creditcard.csv",
        "--threshold-objective", "target-recall",
        "--val-size", "0.2",
        "--target-recall", good,
    ])
    assert 0.0 <= args.target_recall <= 1.0


# ---------------------------------------------------------------------------
# CLI: bad combo --target-recall without --threshold-objective target-recall
# ---------------------------------------------------------------------------


def test_parse_args_rejects_target_recall_without_target_recall_objective():
    with pytest.raises(SystemExit):
        parse_args([
            "--data-path", "data/creditcard.csv",
            "--target-recall", "0.9",
            # --threshold-objective defaults to f1
        ])


def test_parse_args_accepts_target_recall_with_correct_objective():
    args = parse_args([
        "--data-path", "data/creditcard.csv",
        "--threshold-objective", "target-recall",
        "--val-size", "0.2",
        "--target-recall", "0.9",
    ])
    assert args.target_recall == pytest.approx(0.9)
    assert args.threshold_objective == "target-recall"


# ---------------------------------------------------------------------------
# CLI: --val-size required when --threshold-objective != f1 or --target-recall set
# ---------------------------------------------------------------------------


def test_parse_args_rejects_target_recall_objective_without_val_size():
    with pytest.raises(SystemExit):
        parse_args([
            "--data-path", "data/creditcard.csv",
            "--threshold-objective", "target-recall",
            # no --val-size
        ])


def test_parse_args_rejects_target_recall_value_without_val_size():
    with pytest.raises(SystemExit):
        parse_args([
            "--data-path", "data/creditcard.csv",
            "--threshold-objective", "target-recall",
            "--target-recall", "0.9",
            # no --val-size
        ])


def test_parse_args_accepts_target_recall_objective_with_val_size():
    args = parse_args([
        "--data-path", "data/creditcard.csv",
        "--threshold-objective", "target-recall",
        "--val-size", "0.2",
    ])
    assert args.threshold_objective == "target-recall"
    assert args.val_size == pytest.approx(0.2)


def test_parse_args_f1_objective_does_not_require_val_size():
    # default objective is f1; no --val-size should parse fine
    args = parse_args(["--data-path", "data/creditcard.csv"])
    assert args.val_size is None
    assert args.threshold_objective == "f1"


# ---------------------------------------------------------------------------
# FeaturePipeline: schema validation
# ---------------------------------------------------------------------------


def _minimal_df(n: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "Time": rng.integers(0, 86400, n).astype(float),
        "Amount": rng.exponential(scale=10, size=n),
    })


def test_feature_pipeline_fit_raises_on_missing_time():
    df = _minimal_df().drop(columns=["Time"])
    with pytest.raises(ValueError, match="Time"):
        FeaturePipeline().fit(df)


def test_feature_pipeline_fit_raises_on_missing_amount():
    df = _minimal_df().drop(columns=["Amount"])
    with pytest.raises(ValueError, match="Amount"):
        FeaturePipeline().fit(df)


def test_feature_pipeline_transform_raises_on_missing_required_col():
    train = _minimal_df(10)
    pipeline = FeaturePipeline().fit(train)
    bad = _minimal_df(5).drop(columns=["Time"])
    with pytest.raises(ValueError, match="Time"):
        pipeline.transform(bad)


def test_feature_pipeline_works_without_any_v_cols():
    df = _minimal_df(10)
    pipeline = FeaturePipeline().fit(df)
    result = pipeline.transform(df)
    # no V cols present → no V cols in output
    assert not any(col.startswith("V") for col in result.columns)
    assert "log_amount_scaled" in result.columns
    assert "amount_scaled" in result.columns


def test_feature_pipeline_preserves_available_v_cols():
    rng = np.random.default_rng(1)
    n = 10
    df = pd.DataFrame({
        "Time": rng.integers(0, 86400, n).astype(float),
        "Amount": rng.exponential(scale=10, size=n),
        "V1": rng.standard_normal(n),
        "V3": rng.standard_normal(n),  # V2 absent, V3 present
    })
    pipeline = FeaturePipeline().fit(df)
    result = pipeline.transform(df)
    assert "V1" in result.columns
    assert "V3" in result.columns
    assert "V2" not in result.columns


def test_feature_pipeline_transform_raises_if_fit_v_col_missing():
    rng = np.random.default_rng(2)
    n = 10
    train = pd.DataFrame({
        "Time": rng.integers(0, 86400, n).astype(float),
        "Amount": rng.exponential(scale=10, size=n),
        "V1": rng.standard_normal(n),
        "V2": rng.standard_normal(n),
    })
    pipeline = FeaturePipeline().fit(train)
    # transform df missing V2 (was present at fit)
    bad = train.drop(columns=["V2"])
    with pytest.raises(ValueError, match="V2"):
        pipeline.transform(bad)


# ---------------------------------------------------------------------------
# Artifacts: val_threshold only when val_metrics present
# ---------------------------------------------------------------------------


def _base_result(**kwargs) -> TrainingResult:
    return TrainingResult(
        predictions=[],
        training_accuracy=0.99,
        test_accuracy=0.98,
        metrics=ClassificationMetrics(precision=0.8, recall=0.9, f1=0.85, pr_auc=0.92, roc_auc=0.94),
        **kwargs,
    )


def test_val_threshold_omitted_when_val_metrics_absent(tmp_path):
    result = _base_result(val_threshold=0.3)  # val_threshold set, val_metrics absent
    write_artifacts(tmp_path / "run", result=result, config={})
    data = json.loads((tmp_path / "run" / "metrics.json").read_text())
    assert "val_threshold" not in data


def test_val_threshold_present_when_val_metrics_present(tmp_path):
    result = _base_result(
        val_threshold=0.3,
        val_metrics=ClassificationMetrics(precision=0.7, recall=0.88, f1=0.78, pr_auc=0.85, roc_auc=0.90),
    )
    write_artifacts(tmp_path / "run", result=result, config={})
    data = json.loads((tmp_path / "run" / "metrics.json").read_text())
    assert data["val_threshold"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Artifacts: split_counts emitted
# ---------------------------------------------------------------------------


def test_split_counts_in_metrics_two_way(tmp_path):
    result = _base_result(split_counts=SplitCounts(train=800, test=200))
    write_artifacts(tmp_path / "run", result=result, config={})
    data = json.loads((tmp_path / "run" / "metrics.json").read_text())
    assert data["split_train"] == 800
    assert data["split_test"] == 200
    assert "split_val" not in data


def test_split_counts_in_metrics_three_way(tmp_path):
    result = _base_result(split_counts=SplitCounts(train=600, val=200, test=200))
    write_artifacts(tmp_path / "run", result=result, config={})
    data = json.loads((tmp_path / "run" / "metrics.json").read_text())
    assert data["split_train"] == 600
    assert data["split_val"] == 200
    assert data["split_test"] == 200


def test_split_counts_absent_when_none(tmp_path):
    result = _base_result()  # split_counts=None
    write_artifacts(tmp_path / "run", result=result, config={})
    data = json.loads((tmp_path / "run" / "metrics.json").read_text())
    assert "split_train" not in data
    assert "split_val" not in data
    assert "split_test" not in data


# ---------------------------------------------------------------------------
# TrainingResult: split_counts populated by train_one_batch
# ---------------------------------------------------------------------------


def _csv(tmp_path, n_rows: int = 6) -> object:
    path = tmp_path / "data.csv"
    pd.DataFrame({
        "Time": list(range(1, n_rows + 1)),
        "V1": [float(i % 2) * 4 for i in range(n_rows)],
        "Amount": [1.0 + i for i in range(n_rows)],
        "Class": [i % 2 for i in range(n_rows)],
    }).to_csv(path, index=False)
    return path


def test_train_one_batch_populates_split_counts(tmp_path):
    from fraud_detection.models import LightGbmFactory
    from fraud_detection.training import train_one_batch

    result = train_one_batch(
        data_path=_csv(tmp_path),
        model_factory=LightGbmFactory(),
        batch_size=6,
        test_size=0.5,
    )
    assert result.split_counts is not None
    assert result.split_counts.train > 0
    assert result.split_counts.test > 0
    assert result.split_counts.val is None


def test_train_one_batch_three_way_populates_val_split_count(tmp_path):
    from fraud_detection.models import LightGbmFactory
    from fraud_detection.training import train_one_batch

    rng = np.random.default_rng(42)
    n = 20
    path = tmp_path / "data3.csv"
    row: dict = {"Time": list(range(1, n + 1))}
    for i in range(1, 29):
        row[f"V{i}"] = rng.standard_normal(n).tolist()
    row["Amount"] = rng.exponential(10, n).tolist()
    row["Class"] = [i % 2 for i in range(n)]
    pd.DataFrame(row).to_csv(path, index=False)

    result = train_one_batch(
        data_path=path,
        model_factory=LightGbmFactory(),
        batch_size=n,
        test_size=0.2,
        val_size=0.2,
    )
    assert result.split_counts is not None
    assert result.split_counts.train > 0
    assert result.split_counts.val is not None
    assert result.split_counts.val > 0
    assert result.split_counts.test > 0
