from pathlib import Path
import json

from ml_production_ecosystem.production_patterns.quality_gate import evaluate_quality_gate


def test_evaluate_quality_gate_passes_metric_minimums(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"candidate_count": 3, "ratings_rows": 8}))

    result = evaluate_quality_gate(
        {
            "enabled": True,
            "metrics_path": str(metrics_path),
            "minimums": {"candidate_count": 1, "ratings_rows": 1},
        }
    )

    assert result == {"passed": True, "failures": []}


def test_evaluate_quality_gate_reports_failed_minimums(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"candidate_count": 0, "ratings_rows": 8}))

    result = evaluate_quality_gate(
        {
            "enabled": True,
            "metrics_path": str(metrics_path),
            "minimums": {"candidate_count": 1, "ratings_rows": 10},
        }
    )

    assert result == {
        "passed": False,
        "failures": [
            "candidate_count 0.0 below minimum 1.0",
            "ratings_rows 8.0 below minimum 10.0",
        ],
    }


def test_evaluate_quality_gate_disabled_passes_without_checks() -> None:
    assert evaluate_quality_gate({"enabled": False}) == {"passed": True, "failures": []}


def test_evaluate_quality_gate_resolves_relative_metrics_path_against_base_path(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "nested"
    metrics_dir.mkdir()
    metrics_path = metrics_dir / "metrics.json"
    metrics_path.write_text(json.dumps({"accuracy": 0.91}))

    result = evaluate_quality_gate(
        {
            "enabled": True,
            "metrics_path": "nested/metrics.json",
            "minimums": {"accuracy": 0.9},
        },
        base=tmp_path,
    )

    assert result == {"passed": True, "failures": []}


def test_evaluate_quality_gate_minimum_deltas_pass_when_improvement_meets_threshold(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps({"score": 70.0}))
    metrics_path = tmp_path / "candidate.json"
    metrics_path.write_text(json.dumps({"score": 82.0}))

    result = evaluate_quality_gate(
        {
            "enabled": True,
            "metrics_path": str(metrics_path),
            "baseline_metrics_path": str(baseline_path),
            "minimum_deltas": {"score": 5.0},
        }
    )

    assert result == {"passed": True, "failures": []}


def test_evaluate_quality_gate_minimum_deltas_fails_when_improvement_below_threshold(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps({"score": 70.0}))
    metrics_path = tmp_path / "candidate.json"
    metrics_path.write_text(json.dumps({"score": 82.0}))

    result = evaluate_quality_gate(
        {
            "enabled": True,
            "metrics_path": str(metrics_path),
            "baseline_metrics_path": str(baseline_path),
            "minimum_deltas": {"score": 20.0},
        }
    )

    assert result == {
        "passed": False,
        "failures": ["score delta 12.0 below minimum 20.0"],
    }


def test_evaluate_quality_gate_maximum_regressions_pass_within_threshold(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps({"score": 90.0}))
    metrics_path = tmp_path / "candidate.json"
    metrics_path.write_text(json.dumps({"score": 85.0}))

    result = evaluate_quality_gate(
        {
            "enabled": True,
            "metrics_path": str(metrics_path),
            "champion_metrics_path": str(baseline_path),
            "maximum_regressions": {"score": 10.0},
        }
    )

    assert result == {"passed": True, "failures": []}


def test_evaluate_quality_gate_maximum_regressions_fails_when_regression_exceeds_threshold(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps({"score": 90.0}))
    metrics_path = tmp_path / "candidate.json"
    metrics_path.write_text(json.dumps({"score": 75.0}))

    result = evaluate_quality_gate(
        {
            "enabled": True,
            "metrics_path": str(metrics_path),
            "champion_metrics_path": str(baseline_path),
            "maximum_regressions": {"score": 10.0},
        }
    )

    assert result == {
        "passed": False,
        "failures": ["score regression 15.0 above maximum 10.0"],
    }


def test_evaluate_quality_gate_baseline_comparison_evaluates_each_metric_independently(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps({"accuracy": 80.0, "f1": 70.0}))
    metrics_path = tmp_path / "candidate.json"
    metrics_path.write_text(json.dumps({"accuracy": 83.0, "f1": 69.0}))

    result = evaluate_quality_gate(
        {
            "enabled": True,
            "metrics_path": str(metrics_path),
            "baseline_metrics_path": str(baseline_path),
            "minimum_deltas": {"accuracy": 2.0, "f1": 0.0},
        }
    )

    assert result == {
        "passed": False,
        "failures": ["f1 delta -1.0 below minimum 0.0"],
    }
