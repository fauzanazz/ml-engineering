"""Summarize local lifecycle reports for operator status."""

from pathlib import Path
import argparse
import json
from typing import Any

DEFAULT_REPORT_DIR = Path("artifacts/reports/production-patterns")
DEFAULT_OUTPUT_PATH = DEFAULT_REPORT_DIR / "lifecycle-status.json"
REPORT_FILES = {
    "model_contract": ("model-contract-manifest.json",),
    "dataset": ("dataset-manifest.json",),
    "training": ("lifecycle-demo.json", "local-lifecycle-demo.json", "local-deployment-lifecycle.json"),
    "offline_validation": ("offline-validation.json",),
    "approval": ("approval-decision.json",),
    "deployment_demo": ("deployment-demo.json", "local-deployment-demo.json"),
    "drift": ("drift-report.json", "local-deployment-drift.json"),
    "continual_learning": ("continual-learning-decision.json",),
    "continual_learning_summary": ("continual-learning-summary.json",),
    "graph": ("lifecycle-demo.html", "local-lifecycle-demo.html", "local-deployment-lifecycle.html"),
}

def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        return data
    return {}

READY_STATUSES = {"approved", "completed", "monitor", "passed", "ready", "stable"}
BLOCKED_STATUSES = {"blocked", "failed", "pending", "skipped", "unknown"}


def _report_status(path: Path, name: str) -> str:
    if path.suffix == ".html":
        return "ready"
    report = _read_json(path)
    status = report.get("status")
    if name == "training":
        training = report.get("training", {})
        if isinstance(training, dict):
            status = training.get("status", status)
    if name == "continual_learning":
        status = status or report.get("action")
    return str(status or "unknown")


def _status_from_report(report_dir: Path, name: str, filenames: tuple[str, ...]) -> dict[str, object]:
    existing_paths = [report_dir / filename for filename in filenames if (report_dir / filename).exists()]
    if not existing_paths:
        return {"status": "missing", "path": str(report_dir / filenames[0])}
    reports = [(path, _report_status(path, name)) for path in existing_paths]
    ready_report = next(((path, status) for path, status in reports if status in READY_STATUSES), None)
    path, status = ready_report or reports[0]
    return {"status": status, "path": str(path)}

def build_lifecycle_status(
    report_dir: Path = DEFAULT_REPORT_DIR,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, object]:
    steps = {
        name: _status_from_report(report_dir, name, filename)
        for name, filename in REPORT_FILES.items()
    }
    missing = [name for name, step in steps.items() if step["status"] == "missing"]
    blocked = [
        name
        for name, step in steps.items()
        if step["status"] in BLOCKED_STATUSES
    ]
    summary = {
        "status": "ready" if not missing and not blocked else "incomplete",
        "report_dir": str(report_dir),
        "missing_steps": missing,
        "blocked_steps": blocked,
        "steps": steps,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary

def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize local lifecycle report status.")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    summary = build_lifecycle_status(args.report_dir, args.output_path)
    print(json.dumps(summary, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
