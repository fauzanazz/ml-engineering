import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
from lightgbm import LGBMClassifier

from fraud_detection.training import TrainingResult

_JOBLIB_WARNING = (
    "joblib artifacts may execute arbitrary code on load; "
    "only load from trusted sources"
)


def make_run_dir(base: Path, run_id: str | None = None) -> Path:
    if run_id:
        return base / run_id
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    suffix = uuid.uuid4().hex[:8]
    return base / f"{timestamp}-{suffix}"


def write_artifacts(
    run_dir: Path,
    *,
    result: TrainingResult,
    config: dict[str, Any],
    model=None,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=False)

    metrics: dict[str, Any] = {
        "training_accuracy": result.training_accuracy,
        "test_accuracy": result.test_accuracy,
        "precision": result.metrics.precision,
        "recall": result.metrics.recall,
        "f1": result.metrics.f1,
        "pr_auc": result.metrics.pr_auc,
    }
    if result.val_metrics is not None:
        if result.val_threshold is not None:
            metrics["val_threshold"] = result.val_threshold
        metrics["val_precision"] = result.val_metrics.precision
        metrics["val_recall"] = result.val_metrics.recall
        metrics["val_f1"] = result.val_metrics.f1
        metrics["val_pr_auc"] = result.val_metrics.pr_auc

    if result.predict_proba_latency_s is not None:
        metrics["predict_proba_latency_s"] = result.predict_proba_latency_s
        metrics["inference_latency_s"] = result.predict_proba_latency_s  # backward compat alias
    if result.predict_proba_latency_per_row_s is not None:
        metrics["predict_proba_latency_per_row_s"] = result.predict_proba_latency_per_row_s

    if result.split_counts is not None:
        sc = result.split_counts
        metrics["split_train"] = sc.train
        metrics["split_test"] = sc.test
        if sc.val is not None:
            metrics["split_val"] = sc.val

    model_artifact_name: str | None = None
    if model is not None:
        if isinstance(model, LGBMClassifier):
            model_path = run_dir / "model.txt"
            if model_path.exists():
                raise FileExistsError(f"{model_path} already exists")
            model.booster_.save_model(str(model_path))
            model_artifact_name = "model.txt"
        else:
            model_path = run_dir / "model.joblib"
            if model_path.exists():
                raise FileExistsError(f"{model_path} already exists")
            joblib.dump(model, model_path)
            model_artifact_name = "model.joblib"
            metrics["model_artifact_warning"] = _JOBLIB_WARNING

    final_config = dict(config)
    if model_artifact_name is not None:
        final_config["model_artifact"] = model_artifact_name
    if model_artifact_name == "model.joblib":
        final_config["model_artifact_warning"] = _JOBLIB_WARNING

    with open(run_dir / "metrics.json", "x") as f:
        f.write(json.dumps(metrics, indent=2))

    with open(run_dir / "config.json", "x") as f:
        f.write(json.dumps(final_config, indent=2))

    return run_dir
