"""Local canary release decision tests."""

from pathlib import Path
import json
import subprocess
import sys

from production_patterns.canary_decision import build_canary_decision


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def test_canary_decision_promotes_when_evidence_passes(tmp_path: Path) -> None:
    demo = _write_json(tmp_path / "demo.json", {"status": "passed"})
    drift = _write_json(tmp_path / "drift.json", {"status": "passed"})
    approval = _write_json(tmp_path / "approval.json", {"status": "approved"})

    report = build_canary_decision(demo, drift, approval, 10, tmp_path / "canary.json")

    assert report["status"] == "passed"
    assert report["decision"] == "promote"
    assert report["rollback_required"] is False
    assert report["failures"] == []
    assert json.loads((tmp_path / "canary.json").read_text()) == report


def test_canary_decision_blocks_when_drift_fails(tmp_path: Path) -> None:
    demo = _write_json(tmp_path / "demo.json", {"status": "passed"})
    drift = _write_json(tmp_path / "drift.json", {"status": "failed"})

    report = build_canary_decision(demo, drift, None, 20, tmp_path / "canary.json")

    assert report["status"] == "blocked"
    assert report["decision"] == "rollback"
    assert report["rollback_required"] is True
    assert "drift status is failed" in report["failures"]


def test_canary_decision_blocks_invalid_percent(tmp_path: Path) -> None:
    demo = _write_json(tmp_path / "demo.json", {"status": "passed"})
    drift = _write_json(tmp_path / "drift.json", {"status": "passed"})

    report = build_canary_decision(demo, drift, None, 0, tmp_path / "canary.json")

    assert report["status"] == "blocked"
    assert "canary_percent must be between 1 and 100" in report["failures"]


def test_canary_decision_cli_writes_report(tmp_path: Path) -> None:
    demo = _write_json(tmp_path / "demo.json", {"status": "passed"})
    drift = _write_json(tmp_path / "drift.json", {"status": "passed"})
    output_path = tmp_path / "canary.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "production_patterns.canary_decision",
            "--deployment-demo",
            str(demo),
            "--drift-report",
            str(drift),
            "--output-path",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    report = json.loads(output_path.read_text())
    assert report["decision"] == "promote"
    assert "canary_percent" in result.stdout
