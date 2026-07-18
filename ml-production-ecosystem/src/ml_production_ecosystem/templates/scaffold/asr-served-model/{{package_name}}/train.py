from pathlib import Path
import argparse
import json


def write_training_summary(summary_path: Path) -> dict[str, str]:
    summary = {
        "model_name": "{{package_name}}",
        "version": "local-dev",
        "artifact_uri": "models/local-dev",
        "metrics_uri": "reports/metrics.json",
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    metrics_path = Path(summary["metrics_uri"])
    if not metrics_path.is_absolute():
        metrics_path = summary_path.parent.parent / metrics_path
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps({"wer": 0.0, "cer": 0.0}, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Write local ASR training summary.")
    parser.add_argument("--summary-path", type=Path, default=Path("reports/training-summary.json"))
    args = parser.parse_args()
    print(json.dumps(write_training_summary(args.summary_path), sort_keys=True))


if __name__ == "__main__":
    main()
