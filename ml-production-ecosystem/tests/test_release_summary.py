import json
from pathlib import Path
import subprocess
import sys
import tomllib

import yaml

from production_patterns.release_summary import build_release_summary

ROOT = Path(__file__).resolve().parents[1]


def _write_inputs(tmp_path: Path, quality_gate_status: str = "passed", monitor_status: str = "healthy") -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    retraining_report = tmp_path / "scheduled-retraining.json"
    retraining_report.write_text(
        json.dumps(
            {
                "status": "completed",
                "run_id": "foundation-config-v1",
                "model_name": "movielens-popularity",
                "quality_gate": {"status": quality_gate_status},
            }
        )
    )

    deployment_manifest = tmp_path / "deployment-manifest.yaml"
    deployment_manifest.write_text(
        yaml.safe_dump(
            {
                "service_name": "foundation-api",
                "image": "ml-production-ecosystem-foundation-api",
                "command": "uv run foundation-serve-recommender --host 0.0.0.0 --port 8000",
                "port": 8000,
                "health_endpoint": "/health",
            }
        )
    )

    monitor_summary = tmp_path / "monitoring-loop.json"
    monitor_summary.write_text(json.dumps({"status": monitor_status, "checks": []}))

    return {
        "retraining_report": retraining_report,
        "deployment_manifest": deployment_manifest,
        "monitor_summary": monitor_summary,
    }


def test_release_summary_ready_when_quality_smoke_and_monitor_pass(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    output_path = tmp_path / "release-summary.json"

    summary = build_release_summary(
        retraining_report_path=paths["retraining_report"],
        deployment_manifest_path=paths["deployment_manifest"],
        monitor_summary_path=paths["monitor_summary"],
        smoke_status="passed",
        rollback_target="foundation-config-v1",
        output_path=output_path,
    )

    assert summary == {
        "status": "ready",
        "model_name": "movielens-popularity",
        "run_id": "foundation-config-v1",
        "quality_gate": {"status": "passed"},
        "service_name": "foundation-api",
        "image": "ml-production-ecosystem-foundation-api",
        "smoke_status": "passed",
        "monitor_status": "healthy",
        "rollback_target": "foundation-config-v1",
    }
    assert json.loads(output_path.read_text()) == summary


def test_release_summary_blocks_when_smoke_monitor_or_quality_gate_fails(tmp_path: Path) -> None:
    smoke_paths = _write_inputs(tmp_path / "smoke")
    smoke_summary = build_release_summary(
        smoke_paths["retraining_report"],
        smoke_paths["deployment_manifest"],
        smoke_paths["monitor_summary"],
        smoke_status="failed",
        rollback_target="foundation-config-v1",
    )
    assert smoke_summary["status"] == "blocked"

    monitor_paths = _write_inputs(tmp_path / "monitor", monitor_status="unhealthy")
    monitor_summary = build_release_summary(
        monitor_paths["retraining_report"],
        monitor_paths["deployment_manifest"],
        monitor_paths["monitor_summary"],
        smoke_status="passed",
        rollback_target="foundation-config-v1",
    )
    assert monitor_summary["status"] == "blocked"

    quality_paths = _write_inputs(tmp_path / "quality", quality_gate_status="failed")
    quality_summary = build_release_summary(
        quality_paths["retraining_report"],
        quality_paths["deployment_manifest"],
        quality_paths["monitor_summary"],
        smoke_status="passed",
        rollback_target="foundation-config-v1",
    )
    assert quality_summary["status"] == "blocked"


def test_release_summary_handles_missing_file_without_stack_trace(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    missing_path = tmp_path / "missing-monitoring-loop.json"

    summary = build_release_summary(
        paths["retraining_report"],
        paths["deployment_manifest"],
        missing_path,
        smoke_status="passed",
        rollback_target="foundation-config-v1",
    )

    assert summary["status"] == "blocked"
    assert summary["error"] == f"missing input file: {missing_path}"
    assert "Traceback" not in json.dumps(summary)


def test_release_summary_cli_prints_json_and_script_is_registered(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    output_path = tmp_path / "release-summary.json"
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())

    assert pyproject["project"]["scripts"]["production-release-summary"] == "production_patterns.release_summary:main"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "production_patterns.release_summary",
            "--retraining-report",
            str(paths["retraining_report"]),
            "--deployment-manifest",
            str(paths["deployment_manifest"]),
            "--monitor-summary",
            str(paths["monitor_summary"]),
            "--smoke-status",
            "passed",
            "--rollback-target",
            "foundation-config-v1",
            "--output-path",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(result.stdout)
    assert summary["status"] == "ready"
    assert json.loads(output_path.read_text()) == summary
