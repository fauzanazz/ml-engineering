from pathlib import Path
import json

def write_training_summary(summary_path: Path) -> dict[str, str]:
    summary = {"model_name": "{{package_name}}", "version": "local-dev", "artifact_uri": "models/local-dev", "metrics_uri": "reports/metrics.json"}
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    metrics_path = summary_path.parent.parent / "reports" / "metrics.json"
    metrics_path.write_text(json.dumps({"accuracy": 1.0}, indent=2, sort_keys=True) + "\n")
    return summary
