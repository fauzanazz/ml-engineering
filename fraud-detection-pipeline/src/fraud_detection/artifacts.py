import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from fraud_detection.inference import BUNDLE_SCHEMA_VERSION, InferenceBundle
from fraud_detection.training import TrainingResult

SCHEMA_VERSION = 1


def make_run_dir(base: Path, run_id: str | None = None) -> Path:
    if run_id is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        suffix = uuid.uuid4().hex[:8]
        run_id = f"{timestamp}-{suffix}"
    return base / run_id


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_dataset_metadata(path: Path, *, target_column: str) -> dict[str, Any]:
    dataset_path = Path(path)
    digest = hashlib.sha256()
    with dataset_path.open("rb") as raw:
        for chunk in iter(lambda: raw.read(1024 * 1024), b""):
            digest.update(chunk)

    with dataset_path.open(encoding="utf-8", newline="") as text:
        reader = csv.reader(text)
        try:
            columns = next(reader)
        except StopIteration as exc:
            raise ValueError("dataset is empty") from exc
        rows = sum(1 for _ in reader)
    if target_column not in columns:
        raise ValueError(f"dataset target column is missing: {target_column}")

    return {
        "path": str(dataset_path),
        "sha256": digest.hexdigest(),
        "bytes": dataset_path.stat().st_size,
        "rows": rows,
        "columns": columns,
    }


def collect_runtime_metadata() -> dict[str, Any]:
    packages = [
        "joblib",
        "lightgbm",
        "numpy",
        "optuna",
        "pandas",
        "scikit-learn",
        "xgboost",
    ]
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        **{package: importlib.metadata.version(package) for package in packages},
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_artifacts(
    run_dir: Path,
    *,
    result: TrainingResult,
    config: dict[str, Any],
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=False)
    if set(config) != {
        "schema_version",
        "command",
        "dataset",
        "run",
        "runtime",
        "training",
        "split",
        "tuning",
        "threshold",
    }:
        raise ValueError("config must contain the schema-version-1 top-level contract")
    if config["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")

    bundle = InferenceBundle(
        schema_version=BUNDLE_SCHEMA_VERSION,
        model=result.model,
        feature_pipeline=result.feature_pipeline,
        effective_threshold=result.effective_threshold,
        input_columns=result.input_columns,
        model_key=config["training"]["model_key"],
    )
    bundle_path = run_dir / "bundle.joblib"
    evaluation_path = run_dir / "evaluation.npz"
    joblib.dump(bundle, bundle_path)
    np.savez(
        evaluation_path,
        test_labels=result.test_labels,
        test_scores=result.test_scores,
    )

    final_config = dict(config)
    final_config["artifacts"] = {
        "bundle.joblib": {
            "path": "bundle.joblib",
            "sha256": _sha256(bundle_path),
            "format": "joblib",
            "trusted_load_required": True,
        },
        "evaluation.npz": {
            "path": "evaluation.npz",
            "sha256": _sha256(evaluation_path),
        },
    }
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "training_accuracy": result.training_accuracy,
        "test_accuracy": result.test_accuracy,
        "validation": asdict(result.val_metrics) if result.val_metrics is not None else None,
        "test": asdict(result.metrics),
        "latency": {
            "batch_s": result.predict_proba_latency_s,
            "per_row_s": result.predict_proba_latency_per_row_s,
            "single_row_s": result.single_row_latency_s,
            "warmup_calls": 1,
            "single_row_repeats": 5,
            "statistic": "median",
        },
    }

    with (run_dir / "config.json").open("x", encoding="utf-8") as file:
        json.dump(_json_safe(final_config), file, indent=2, sort_keys=True)
        file.write("\n")
    with (run_dir / "metrics.json").open("x", encoding="utf-8") as file:
        json.dump(_json_safe(metrics), file, indent=2, sort_keys=True)
        file.write("\n")
    return run_dir
