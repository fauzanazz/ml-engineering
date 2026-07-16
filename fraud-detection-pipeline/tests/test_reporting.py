import json

import numpy as np
import pandas as pd
import pytest

from fraud_detection.cli import main as train_main
from fraud_detection.reporting import generate_full_dataset_report


def _run(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "Time": np.arange(24, dtype=float),
            "Amount": np.resize([1.0, 100.0], 24),
            "V1": np.resize([0.0, 4.0], 24),
            "Class": np.resize([0, 1], 24),
        }
    ).to_csv(path, index=False)
    artifact_dir = tmp_path / "runs"
    train_main(
        [
            "--data-path",
            str(path),
            "--batch-size",
            "24",
            "--test-size",
            "0.25",
            "--val-size",
            "0.25",
            "--threshold-objective",
            "target-recall",
            "--target-recall",
            "0.95",
            "--model",
            "logistic-regression",
            "--seed",
            "7",
            "--artifact-dir",
            str(artifact_dir),
        ]
    )
    return next(artifact_dir.iterdir())


def test_report_generates_exact_markdown_and_svg_outputs(tmp_path):
    run_dir = _run(tmp_path)
    output_dir = tmp_path / "report"

    report, pr_curve, confusion = generate_full_dataset_report(run_dir, output_dir)

    assert report == output_dir / "full-dataset-benchmark.md"
    assert pr_curve == output_dir / "assets" / "full-dataset-pr-curve.svg"
    assert confusion == output_dir / "assets" / "full-dataset-confusion-matrix.svg"
    assert {path.relative_to(output_dir).as_posix() for path in output_dir.rglob("*") if path.is_file()} == {
        "full-dataset-benchmark.md",
        "assets/full-dataset-pr-curve.svg",
        "assets/full-dataset-confusion-matrix.svg",
    }
    assert pr_curve.read_text().lstrip().startswith("<?xml")
    assert confusion.read_text().lstrip().startswith("<?xml")


def test_report_values_match_verified_json_and_npz(tmp_path):
    run_dir = _run(tmp_path)
    report_path, _, _ = generate_full_dataset_report(run_dir, tmp_path / "report")
    report = report_path.read_text()
    config = json.loads((run_dir / "config.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())

    assert config["dataset"]["sha256"] in report
    assert f"Dataset rows: {config['dataset']['rows']}" in report
    assert f"Selected threshold: {config['threshold']['selected']:.12f}" in report
    assert f"| True positives | {metrics['test']['true_positives']} |" in report
    assert f"| False negatives | {metrics['test']['false_negatives']} |" in report
    assert "held-out test" in report.lower()
    assert "The dataset is from 2013" in report
    with np.load(run_dir / "evaluation.npz") as evaluation:
        assert len(evaluation["test_labels"]) == (
            metrics["test"]["positive_support"] + metrics["test"]["negative_support"]
        )


def test_report_markdown_is_deterministic_for_one_run(tmp_path):
    run_dir = _run(tmp_path)
    first, _, _ = generate_full_dataset_report(run_dir, tmp_path / "first")
    second, _, _ = generate_full_dataset_report(run_dir, tmp_path / "second")
    assert first.read_text() == second.read_text()


def test_report_rejects_schema_and_evaluation_hash_mismatch(tmp_path):
    run_dir = _run(tmp_path)
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text())
    config["schema_version"] = 2
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="schema_version"):
        generate_full_dataset_report(run_dir, tmp_path / "schema-report")

    config["schema_version"] = 1
    config_path.write_text(json.dumps(config))
    with (run_dir / "evaluation.npz").open("ab") as evaluation:
        evaluation.write(b"corrupt")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        generate_full_dataset_report(run_dir, tmp_path / "hash-report")


def test_report_refuses_to_overwrite_existing_evidence(tmp_path):
    run_dir = _run(tmp_path)
    output_dir = tmp_path / "report"
    generate_full_dataset_report(run_dir, output_dir)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        generate_full_dataset_report(run_dir, output_dir)
