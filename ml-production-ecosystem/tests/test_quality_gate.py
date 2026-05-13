from pathlib import Path
import json

from production_patterns.quality_gate import evaluate_quality_gate


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
