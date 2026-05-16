"""Local scheduler plan validation tests."""

from pathlib import Path
import json
import subprocess
import sys

import yaml

from ml_production_ecosystem.production_patterns.local_scheduler import run_local_scheduler, validate_local_scheduler

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "platform" / "local" / "scheduler" / "jobs.yaml"


def test_local_scheduler_plan_passes(tmp_path: Path) -> None:
    report = validate_local_scheduler(PLAN, tmp_path / "scheduler.json")

    assert report["status"] == "passed"
    assert report["provider"] == "local"
    assert report["runtime"] == "cron-compatible"
    assert report["job_count"] == 2
    assert report["jobs"][0]["name"] == "scheduled-retraining"
    assert report["failures"] == []
    assert json.loads((tmp_path / "scheduler.json").read_text()) == report


def test_local_scheduler_rejects_non_production_command(tmp_path: Path) -> None:
    plan = yaml.safe_load(PLAN.read_text())
    plan["jobs"][0]["command"] = ["python", "script.py", "run"]
    bad_plan = tmp_path / "jobs.yaml"
    bad_plan.write_text(yaml.safe_dump(plan))

    report = validate_local_scheduler(bad_plan, tmp_path / "scheduler.json")

    assert report["status"] == "failed"
    assert "jobs[0].command must start with uv run" in report["failures"]


def test_local_scheduler_rejects_bad_output_path(tmp_path: Path) -> None:
    plan = yaml.safe_load(PLAN.read_text())
    plan["jobs"][0]["output_path"] = "/tmp/out.json"
    bad_plan = tmp_path / "jobs.yaml"
    bad_plan.write_text(yaml.safe_dump(plan))

    report = validate_local_scheduler(bad_plan, tmp_path / "scheduler.json")

    assert report["status"] == "failed"
    assert "jobs[0].output_path must stay under production reports" in report["failures"]


def test_local_scheduler_cli_writes_report(tmp_path: Path) -> None:
    output = tmp_path / "scheduler.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ml_production_ecosystem.production_patterns.local_scheduler",
            "--plan-path",
            str(PLAN),
            "--output-path",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    report = json.loads(output.read_text())
    assert report["status"] == "passed"
    assert "scheduled-retraining" in result.stdout

def test_local_scheduler_runtime_plans_jobs_without_managed_service(tmp_path: Path) -> None:
    report = run_local_scheduler(PLAN, "lifecycle-status", True, tmp_path / "scheduler-run.json")

    assert report["status"] == "passed"
    assert report["mode"] == "local-scheduler-runtime"
    assert report["dry_run"] is True
    assert report["job_count"] == 1
    assert report["runs"][0]["name"] == "lifecycle-status"
    assert report["runs"][0]["status"] == "planned"
    assert report["runs"][0]["command"][:3] == ["uv", "run", "production-lifecycle-status"]
    assert json.loads((tmp_path / "scheduler-run.json").read_text()) == report

def test_local_scheduler_runtime_reports_missing_job(tmp_path: Path) -> None:
    report = run_local_scheduler(PLAN, "missing", True, tmp_path / "scheduler-run.json")

    assert report["status"] == "failed"
    assert "job not found: missing" in report["failures"]

def test_local_scheduler_runtime_cli_writes_report(tmp_path: Path) -> None:
    output = tmp_path / "scheduler-run.json"
    result = subprocess.run(
        [
            "uv",
            "run",
            "production-run-local-scheduler",
            "--plan-path",
            str(PLAN),
            "--job-name",
            "lifecycle-status",
            "--output-path",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    report = json.loads(output.read_text())
    assert report["mode"] == "local-scheduler-runtime"
    assert report["runs"][0]["status"] == "planned"
    assert "local-scheduler-runtime" in result.stdout
