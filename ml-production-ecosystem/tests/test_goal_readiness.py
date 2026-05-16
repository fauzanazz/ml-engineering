"""Goal readiness audit tests."""

from pathlib import Path
import json
import subprocess
import sys

from ml_production_ecosystem.production_patterns.goal_readiness import build_goal_readiness_report

ROOT = Path(__file__).resolve().parents[1]


def test_goal_readiness_reports_current_progress_and_gaps(tmp_path: Path) -> None:
    report = build_goal_readiness_report(ROOT, tmp_path / "goal.json")

    assert report["status"] == "ready"
    assert report["completion_claim"] == "complete_candidate"
    assert report["failed_checks"] == []
    assert "readme_vision" in report["passed_checks"]
    assert "cloud_provider_adapters" in report["passed_checks"]
    assert "model_contracts_reusable" in report["passed_checks"]
    assert "local_canary_workflow" in report["passed_checks"]
    assert "local_canary_router" in report["passed_checks"]
    assert "slo_burn_rate_simulation" in report["passed_checks"]
    assert "multi_window_burn_rate_alerting" in report["passed_checks"]
    assert "autoscaling_decision_simulation" in report["passed_checks"]
    assert "distributed_load_aggregation" in report["passed_checks"]
    assert "local_cost_estimation" in report["passed_checks"]
    assert "local_kubernetes_parity" in report["passed_checks"]
    assert "local_scheduler_plan" in report["passed_checks"]
    assert report["known_gaps"] == []
    assert json.loads((tmp_path / "goal.json").read_text()) == report


def test_goal_readiness_fails_when_evidence_missing(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("local-first model-agnostic provider-agnostic")

    report = build_goal_readiness_report(tmp_path, tmp_path / "goal.json")

    assert report["status"] == "in_progress"
    assert "local_lifecycle_without_cloud" in report["failed_checks"]
    assert any(check["status"] == "failed" for check in report["checks"])


def test_goal_readiness_cli_writes_report(tmp_path: Path) -> None:
    output_path = tmp_path / "goal.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ml_production_ecosystem.production_patterns.goal_readiness",
            "--root",
            str(ROOT),
            "--output-path",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    report = json.loads(output_path.read_text())
    assert report["completion_claim"] == "complete_candidate"
    assert "known_gaps" in result.stdout
