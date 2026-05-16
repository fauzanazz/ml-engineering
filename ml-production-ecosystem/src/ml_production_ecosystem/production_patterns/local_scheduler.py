"""Validate and run local scheduler jobs without managed services."""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PLAN_PATH = Path("configs/platform/local/scheduler/jobs.yaml")
DEFAULT_OUTPUT_PATH = Path("artifacts/reports/production-patterns/local-scheduler-validation.json")
DEFAULT_RUN_OUTPUT_PATH = Path("artifacts/reports/production-patterns/local-scheduler-run.json")
ALLOWED_COMMAND_PREFIX = ("uv", "run")


def validate_local_scheduler(
    plan_path: Path = DEFAULT_PLAN_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    plan = _read_yaml(plan_path)
    jobs = _items(plan.get("jobs", []))
    failures = []

    if plan.get("provider") != "local":
        failures.append("provider must be local")
    if plan.get("runtime") != "cron-compatible":
        failures.append("runtime must be cron-compatible")
    if not jobs:
        failures.append("at least one job is required")

    seen_names: set[str] = set()
    for index, job in enumerate(jobs):
        failures.extend(_job_failures(index, job, seen_names))

    report: dict[str, Any] = {
        "status": "passed" if not failures else "failed",
        "plan_path": str(plan_path),
        "provider": plan.get("provider"),
        "runtime": plan.get("runtime"),
        "job_count": len(jobs),
        "jobs": [_job_summary(job) for job in jobs],
        "failures": failures,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report

def run_local_scheduler(
    plan_path: Path = DEFAULT_PLAN_PATH,
    job_name: str | None = None,
    dry_run: bool = True,
    output_path: Path = DEFAULT_RUN_OUTPUT_PATH,
) -> dict[str, Any]:
    validation = validate_local_scheduler(plan_path, output_path.with_suffix(".validation.json"))
    selected_jobs = [
        job for job in validation["jobs"]
        if job_name is None or job["name"] == job_name
    ]
    failures = list(validation["failures"])
    if job_name is not None and not selected_jobs:
        failures.append(f"job not found: {job_name}")

    runs = []
    if not failures:
        runs = [_run_job(job, dry_run) for job in selected_jobs]
        failures.extend(run["error"] for run in runs if run["status"] == "failed")

    report: dict[str, Any] = {
        "status": "passed" if not failures else "failed",
        "mode": "local-scheduler-runtime",
        "dry_run": dry_run,
        "plan_path": str(plan_path),
        "job_name": job_name,
        "job_count": len(selected_jobs),
        "runs": runs,
        "failures": failures,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _job_failures(index: int, job: dict[str, Any], seen_names: set[str]) -> list[str]:
    failures = []
    name = str(job.get("name", ""))
    if not name:
        failures.append(f"jobs[{index}].name is required")
    elif name in seen_names:
        failures.append(f"duplicate job name: {name}")
    seen_names.add(name)

    schedule = str(job.get("schedule", ""))
    if len(schedule.split()) != 5:
        failures.append(f"jobs[{index}].schedule must be 5-field cron expression")

    command = job.get("command", [])
    if not isinstance(command, list) or len(command) < 3:
        failures.append(f"jobs[{index}].command must be non-empty argv list")
    else:
        if tuple(command[:2]) != ALLOWED_COMMAND_PREFIX:
            failures.append(f"jobs[{index}].command must start with uv run")
        if not str(command[2]).startswith("production-"):
            failures.append(f"jobs[{index}].command must run production CLI")

    output_path = str(job.get("output_path", ""))
    if not output_path.startswith("artifacts/reports/production-patterns/"):
        failures.append(f"jobs[{index}].output_path must stay under production reports")
    return failures


def _job_summary(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": str(job.get("name", "")),
        "schedule": str(job.get("schedule", "")),
        "command": [str(part) for part in _items(job.get("command", []))],
        "output_path": str(job.get("output_path", "")),
    }

def _run_job(job: dict[str, Any], dry_run: bool) -> dict[str, Any]:
    command = [str(part) for part in job["command"]]
    if dry_run:
        return {
            "name": job["name"],
            "status": "planned",
            "command": command,
            "output_path": job["output_path"],
            "returncode": None,
            "error": "",
        }

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    status = "passed" if result.returncode == 0 else "failed"
    return {
        "name": job["name"],
        "status": status,
        "command": command,
        "output_path": job["output_path"],
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
        "error": "" if status == "passed" else f"{job['name']} exited {result.returncode}",
    }


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if isinstance(data, dict):
        return data
    return {}


def _items(value: object) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate or run local scheduler jobs without managed services.")
    parser.add_argument("--plan-path", type=Path, default=DEFAULT_PLAN_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--run", action="store_true", help="Run scheduler jobs now instead of validation only.")
    parser.add_argument("--job-name")
    parser.add_argument("--execute", action="store_true", help="Execute commands; default run mode is dry-run.")
    args = parser.parse_args()

    if args.run:
        report = run_local_scheduler(
            plan_path=args.plan_path,
            job_name=args.job_name,
            dry_run=not args.execute,
            output_path=args.output_path,
        )
    else:
        report = validate_local_scheduler(args.plan_path, args.output_path)
    print(json.dumps(report, indent=2, sort_keys=True))


def run_main() -> None:
    parser = argparse.ArgumentParser(description="Run local scheduler jobs without managed services.")
    parser.add_argument("--plan-path", type=Path, default=DEFAULT_PLAN_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_RUN_OUTPUT_PATH)
    parser.add_argument("--job-name")
    parser.add_argument("--execute", action="store_true", help="Execute commands; default is dry-run.")
    args = parser.parse_args()
    report = run_local_scheduler(
        plan_path=args.plan_path,
        job_name=args.job_name,
        dry_run=not args.execute,
        output_path=args.output_path,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
