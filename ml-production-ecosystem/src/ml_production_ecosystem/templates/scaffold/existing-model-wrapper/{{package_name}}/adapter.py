from pathlib import Path
import json
import subprocess

def run_command(command: list[str]) -> None:
    if command:
        subprocess.run(command, check=True)

def write_summary(summary_path: Path, metrics: dict[str, float] | None = None) -> dict[str, str]:
    summary = {"model_name": "{{package_name}}", "version": "external", "artifact_uri": "models/external", "metrics_uri": "reports/metrics.json"}
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    metrics_path = summary_path.parent.parent / "reports" / "metrics.json"
    metrics_path.write_text(json.dumps(metrics or {}, indent=2, sort_keys=True) + "\n")
    return summary
