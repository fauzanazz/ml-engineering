import argparse
import hashlib
import json
import shlex
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, precision_score, recall_score

SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version in {path.name}")
    return value


def _status(value: bool | None) -> str:
    if value is None:
        return "n/a"
    return str(value).lower()


def _metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def generate_full_dataset_report(
    run_dir: Path, output_dir: Path
) -> tuple[Path, Path, Path]:
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    report_path = output_dir / "full-dataset-benchmark.md"
    assets_dir = output_dir / "assets"
    pr_path = assets_dir / "full-dataset-pr-curve.svg"
    confusion_path = assets_dir / "full-dataset-confusion-matrix.svg"
    for path in (report_path, pr_path, confusion_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing report evidence: {path}")

    config = _load_json(run_dir / "config.json")
    metrics = _load_json(run_dir / "metrics.json")
    manifest = config.get("artifacts", {}).get("evaluation.npz")
    if not isinstance(manifest, dict):
        raise ValueError("evaluation.npz manifest is missing")
    evaluation_path = run_dir / manifest.get("path", "")
    if _sha256(evaluation_path) != manifest.get("sha256"):
        raise ValueError("evaluation.npz SHA-256 mismatch")
    with np.load(evaluation_path, allow_pickle=False) as evaluation:
        labels = evaluation["test_labels"]
        scores = evaluation["test_scores"]

    threshold = float(config["threshold"]["selected"])
    predictions = (scores >= threshold).astype(int)
    precision, recall, _ = precision_recall_curve(labels, scores)
    operating_precision = precision_score(labels, predictions, zero_division=0)
    operating_recall = recall_score(labels, predictions, zero_division=0)
    prevalence = float(np.mean(labels))

    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(recall, precision, label="Held-out test PR curve")
    axis.axhline(prevalence, linestyle="--", color="gray", label=f"Prevalence ({prevalence:.6f})")
    axis.scatter(
        [operating_recall],
        [operating_precision],
        color="red",
        zorder=3,
        label=f"Selected threshold ({threshold:.6f})",
    )
    axis.set(xlabel="Recall", ylabel="Precision", title="Held-out test precision-recall curve")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(pr_path, format="svg", metadata={"Date": None})
    plt.close(figure)

    test_metrics = metrics["test"]
    confusion = np.array(
        [
            [test_metrics["true_negatives"], test_metrics["false_positives"]],
            [test_metrics["false_negatives"], test_metrics["true_positives"]],
        ],
        dtype=int,
    )
    figure, axis = plt.subplots(figsize=(5, 5))
    image = axis.imshow(confusion, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(confusion[row, column]), ha="center", va="center")
    axis.set_xticks([0, 1], labels=["Predicted 0", "Predicted 1"])
    axis.set_yticks([0, 1], labels=["Actual 0", "Actual 1"])
    axis.set_title("Held-out test confusion matrix")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(confusion_path, format="svg", metadata={"Date": None})
    plt.close(figure)

    dataset = config["dataset"]
    runtime = config["runtime"]
    training = config["training"]
    split = config["split"]
    tuning = config["tuning"]
    threshold_config = config["threshold"]
    latency = metrics["latency"]
    audit_lines = []
    for name in ("train", "val", "test"):
        partition = split["audit"].get(name)
        if partition is None:
            continue
        audit_lines.append(
            f"| {name} | {partition['rows']} | {partition['positives']} | "
            f"{partition['negatives']} | {partition['time_min']:.6f} | {partition['time_max']:.6f} |"
        )
    tuning_protocol = (
        f"{tuning['n_trials']} Optuna TPE trials; {tuning['scoring']}; "
        f"{tuning['cv_strategy']} with {tuning['cv_splits']} effective folds; "
        f"best CV score {tuning['best_score']:.6f}; params `{json.dumps(tuning['best_params'], sort_keys=True)}`"
        if tuning["enabled"]
        else "Disabled"
    )
    runtime_versions = ", ".join(
        f"{name} {version}"
        for name, version in runtime.items()
        if name not in {"python", "platform"}
    )
    command = shlex.join(config["command"])
    report = f"""# Full-Dataset Fraud Detection Benchmark

## Reproduction

- Command: `{command}`
- Dataset: `{dataset['path']}`
- Dataset rows: {dataset['rows']}
- Dataset columns: {len(dataset['columns'])}
- Dataset bytes: {dataset['bytes']}
- Dataset SHA-256: `{dataset['sha256']}`
- Started (UTC): {config['run']['started_at_utc']}
- Training duration: {config['run']['duration_s']:.6f} s
- Seed: {config['run']['seed']}

## Runtime

- Python: {runtime['python']}
- Platform: {runtime['platform']}
- Packages: {runtime_versions}

## Model and tuning protocol

- Model: `{training['model_key']}` ({training['estimator_class']})
- Batch size: {training['batch_size']}
- Imbalance strategy: `{training['imbalance_strategy']}`
- Tuning: {tuning_protocol}
- Chronological split: validation={split['val_size']}, held-out test={split['test_size']}

| Partition | Rows | Positives | Negatives | Time min | Time max |
|---|---:|---:|---:|---:|---:|
{chr(10).join(audit_lines)}

## Selected operating threshold

- Objective: `{threshold_config['objective']}`
- Target recall: {_metric(threshold_config['target_recall'])}
- Selected threshold: {threshold:.12f}
- Target met: {_status(threshold_config['target_met'])}
- F1 fallback used: {_status(threshold_config['fallback_used'])}

## Held-out test results

| Metric | Value |
|---|---:|
| Precision | {_metric(test_metrics['precision'])} |
| Recall | {_metric(test_metrics['recall'])} |
| F1 | {_metric(test_metrics['f1'])} |
| PR AUC | {_metric(test_metrics['pr_auc'])} |
| ROC AUC | {_metric(test_metrics['roc_auc'])} |
| True positives | {test_metrics['true_positives']} |
| True negatives | {test_metrics['true_negatives']} |
| False positives | {test_metrics['false_positives']} |
| False negatives | {test_metrics['false_negatives']} |
| Positive support | {test_metrics['positive_support']} |
| Negative support | {test_metrics['negative_support']} |

![Precision-recall curve](assets/full-dataset-pr-curve.svg)

![Confusion matrix](assets/full-dataset-confusion-matrix.svg)

## Inference latency

- Batch `predict_proba`: {_metric(latency['batch_s'])} s
- Per row: {_metric(latency['per_row_s'])} s
- Single row: {_metric(latency['single_row_s'])} s
- Protocol: {latency['warmup_calls']} warmup call; {latency['single_row_repeats']} single-row repeats; {latency['statistic']} statistic
- Environment: {runtime['platform']}; Python {runtime['python']}

## Limitations

- The dataset is from 2013 and may not represent current fraud patterns.
- V1-V28 are anonymized PCA features, limiting feature-level interpretation.
- Evaluation uses one held-out temporal split rather than repeated production backtests.
- No drift or feedback monitoring is implemented.
- No deployed serving path is included.
- Latency is machine-specific and is not a production service-level guarantee.
"""
    report_path.write_text(report, encoding="utf-8")
    return report_path, pr_path, confusion_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        paths = generate_full_dataset_report(args.run_dir, args.output_dir)
    except (OSError, KeyError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    for path in paths:
        print(path)
