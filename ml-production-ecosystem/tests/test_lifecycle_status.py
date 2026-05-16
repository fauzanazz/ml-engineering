from pathlib import Path
import json

from ml_production_ecosystem.production_patterns.lifecycle_status import build_lifecycle_status

def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload))

def test_lifecycle_status_reports_missing_steps(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    _write_json(report_dir / "model-contract-manifest.json", {"status": "ready"})

    summary = build_lifecycle_status(report_dir, tmp_path / "status.json")

    assert summary["status"] == "incomplete"
    assert "dataset" in summary["missing_steps"]
    assert summary["steps"]["model_contract"]["status"] == "ready"

def test_lifecycle_status_reports_ready_when_all_reports_pass(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    reports = {
        "model-contract-manifest.json": {"status": "ready"},
        "dataset-manifest.json": {"status": "ready"},
        "lifecycle-demo.json": {"training": {"status": "completed"}},
        "offline-validation.json": {"status": "passed"},
        "approval-decision.json": {"status": "approved"},
        "deployment-demo.json": {"status": "passed"},
        "drift-report.json": {"status": "passed"},
        "continual-learning-decision.json": {"status": "monitor"},
        "continual-learning-summary.json": {"status": "stable"},
    }
    for filename, payload in reports.items():
        _write_json(report_dir / filename, payload)
    (report_dir / "lifecycle-demo.html").write_text("<html></html>")

    summary = build_lifecycle_status(report_dir, tmp_path / "status.json")

    assert summary["status"] == "ready"
    assert summary["missing_steps"] == []
    assert summary["blocked_steps"] == []
    assert summary["steps"]["training"]["status"] == "completed"
    assert json.loads((tmp_path / "status.json").read_text()) == summary

def test_lifecycle_status_reports_blocked_failed_report(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    _write_json(report_dir / "drift-report.json", {"status": "failed"})

    summary = build_lifecycle_status(report_dir, tmp_path / "status.json")

    assert summary["status"] == "incomplete"
    assert "drift" in summary["blocked_steps"]

def test_lifecycle_status_reads_local_smoke_report_names(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    reports = {
        "model-contract-manifest.json": {"status": "ready"},
        "dataset-manifest.json": {"status": "ready"},
        "local-lifecycle-demo.json": {"training": {"status": "completed"}},
        "offline-validation.json": {"status": "passed"},
        "approval-decision.json": {"status": "approved"},
        "local-deployment-demo.json": {"status": "passed"},
        "local-deployment-drift.json": {"status": "passed"},
        "continual-learning-decision.json": {"status": "monitor"},
        "continual-learning-summary.json": {"status": "stable"},
    }
    for filename, payload in reports.items():
        _write_json(report_dir / filename, payload)
    (report_dir / "local-lifecycle-demo.html").write_text("<html></html>")

    summary = build_lifecycle_status(report_dir, tmp_path / "status.json")

    assert summary["status"] == "ready"
    assert summary["steps"]["training"]["path"].endswith("local-lifecycle-demo.json")
    assert summary["steps"]["graph"]["path"].endswith("local-lifecycle-demo.html")
