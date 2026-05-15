"""Tiny command-trained classifier proving model-agnostic lifecycle path."""

from pathlib import Path
import argparse
import csv
import json

def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))

def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

def train_classifier(
    data_path: Path,
    artifact_dir: Path,
    summary_path: Path,
    version: str,
) -> dict[str, object]:
    rows = _rows(data_path)
    positive_scores = [float(row["score"]) for row in rows if int(row["label"]) == 1]
    threshold = min(positive_scores)
    correct = 0
    for row in rows:
        prediction = 1 if float(row["score"]) >= threshold else 0
        if prediction == int(row["label"]):
            correct += 1

    model_name = "tiny-threshold-classifier"
    model_dir = artifact_dir / model_name / version
    metrics_path = model_dir / "metrics.json"
    accuracy = correct / len(rows)
    model = {
        "model_name": model_name,
        "version": version,
        "threshold": round(threshold, 6),
        "prediction_key": "label",
    }
    metadata = {
        "model_name": model_name,
        "version": version,
        "framework": "python-standard-library",
        "data_path": str(data_path),
    }
    metrics = {"accuracy": round(accuracy, 6), "rows": len(rows)}
    summary = {
        "model_name": model_name,
        "version": version,
        "artifact_uri": str(model_dir),
        "metrics_uri": str(metrics_path),
    }

    _write_json(model_dir / "model.json", model)
    _write_json(model_dir / "metadata.json", metadata)
    _write_json(metrics_path, metrics)
    _write_json(summary_path, summary)
    return summary

def main() -> None:
    parser = argparse.ArgumentParser(description="Train tiny generic classifier.")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()

    summary = train_classifier(args.data_path, args.artifact_dir, args.summary_path, args.version)
    print(json.dumps(summary, sort_keys=True))

if __name__ == "__main__":
    main()
