import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fraud_detection.training import TrainingResult


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

    metrics = {
        "training_accuracy": result.training_accuracy,
        "test_accuracy": result.test_accuracy,
        "precision": result.metrics.precision,
        "recall": result.metrics.recall,
        "f1": result.metrics.f1,
        "pr_auc": result.metrics.pr_auc,
    }

    with open(run_dir / "metrics.json", "x") as f:
        f.write(json.dumps(metrics, indent=2))

    with open(run_dir / "config.json", "x") as f:
        f.write(json.dumps(config, indent=2))

    if model is not None:
        model_path = run_dir / "model.txt"
        if model_path.exists():
            raise FileExistsError(f"{model_path} already exists")
        model.booster_.save_model(str(model_path))

    return run_dir
