import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from house_prices.training import TrainingResult

_JOBLIB_WARNING = "joblib artifacts may execute arbitrary code on load; only load from trusted sources"


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
    submission: pd.DataFrame | None = None,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=False)

    (run_dir / "metrics.json").write_text(
        json.dumps(asdict(result.metrics), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (run_dir / "artifact-warning.txt").write_text(_JOBLIB_WARNING + "\n", encoding="utf-8")
    joblib.dump(result.model, run_dir / "model.joblib")
    joblib.dump(result.feature_pipeline, run_dir / "feature_pipeline.joblib")

    if submission is not None:
        submission.to_csv(run_dir / "submission.csv", index=False)

    return run_dir
