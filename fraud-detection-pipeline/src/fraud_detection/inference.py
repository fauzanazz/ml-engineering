import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from fraud_detection.features import FeaturePipeline
from fraud_detection.thresholds import apply_threshold

BUNDLE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class InferenceBundle:
    schema_version: int
    model: Any
    feature_pipeline: FeaturePipeline
    effective_threshold: float
    input_columns: tuple[str, ...]
    model_key: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_bundle(run_dir: Path | str, *, trusted: bool = False) -> InferenceBundle:
    if not trusted:
        raise ValueError("Refusing to load pickle-based artifact without trusted=True")

    run_path = Path(run_dir)
    with (run_path / "config.json").open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    if config.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported config schema_version: {config.get('schema_version')!r}"
        )

    manifest = config.get("artifacts", {}).get("bundle.joblib")
    if not isinstance(manifest, dict):
        raise ValueError("config artifacts.bundle.joblib manifest is missing")
    bundle_path = run_path / manifest.get("path", "")
    expected_hash = manifest.get("sha256")
    if not expected_hash or _sha256(bundle_path) != expected_hash:
        raise ValueError("bundle.joblib SHA-256 mismatch")

    bundle = joblib.load(bundle_path)
    if not isinstance(bundle, InferenceBundle):
        raise ValueError("bundle.joblib does not contain an InferenceBundle")
    if bundle.schema_version != BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported bundle schema_version: {bundle.schema_version!r}"
        )
    return bundle


def score_transactions(
    bundle: InferenceBundle, raw_features: pd.DataFrame
) -> pd.DataFrame:
    if raw_features.empty:
        raise ValueError("raw_features must contain at least one row")

    expected = set(bundle.input_columns)
    actual = set(raw_features.columns)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        raise ValueError(f"missing input columns: {missing}")
    if extra:
        raise ValueError(f"extra input columns: {extra}")

    ordered = raw_features.loc[:, list(bundle.input_columns)]
    nonnumeric = [
        column
        for column in ordered.columns
        if not is_numeric_dtype(ordered[column]) or is_bool_dtype(ordered[column])
    ]
    if nonnumeric:
        raise ValueError(f"nonnumeric input columns: {nonnumeric}")
    if not np.isfinite(ordered.to_numpy(dtype=float)).all():
        raise ValueError("input contains non-finite values")

    transformed = bundle.feature_pipeline.transform(ordered)
    probabilities = bundle.model.predict_proba(transformed)[:, 1]
    predictions = apply_threshold(
        probabilities, threshold=bundle.effective_threshold
    )
    return pd.DataFrame(
        {
            "fraud_probability": probabilities,
            "decision_threshold": bundle.effective_threshold,
            "prediction": predictions.astype(int),
        },
        index=raw_features.index,
    )
