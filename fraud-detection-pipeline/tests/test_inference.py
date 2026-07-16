import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from fraud_detection.features import FeaturePipeline
from fraud_detection.inference import InferenceBundle, score_transactions
from fraud_detection.inference import load_bundle
from fraud_detection.cli import main


def _bundle() -> tuple[InferenceBundle, pd.DataFrame]:
    raw = pd.DataFrame(
        {
            "Time": [1.0, 2.0, 3.0, 4.0],
            "Amount": [1.0, 100.0, 2.0, 120.0],
            "V1": [0.0, 3.0, 0.1, 4.0],
        },
        index=[10, 20, 30, 40],
    )
    target = pd.Series([0, 1, 0, 1], index=raw.index)
    pipeline = FeaturePipeline().fit(raw)
    model = LogisticRegression(random_state=42).fit(pipeline.transform(raw), target)
    return (
        InferenceBundle(
            schema_version=1,
            model=model,
            feature_pipeline=pipeline,
            effective_threshold=0.4,
            input_columns=tuple(raw.columns),
            model_key="logistic-regression",
        ),
        raw,
    )


def test_score_transactions_preserves_order_and_exact_output_schema():
    bundle, raw = _bundle()
    scored = score_transactions(bundle, raw.iloc[::-1])

    assert scored.index.tolist() == [40, 30, 20, 10]
    assert scored.columns.tolist() == [
        "fraud_probability",
        "decision_threshold",
        "prediction",
    ]
    assert scored["decision_threshold"].tolist() == [0.4] * 4
    assert scored["prediction"].dtype.kind in "iu"


def test_score_transactions_matches_direct_preprocessing_and_model():
    bundle, raw = _bundle()
    scored = score_transactions(bundle, raw)
    expected = bundle.model.predict_proba(bundle.feature_pipeline.transform(raw))[:, 1]

    assert scored["fraud_probability"].to_numpy() == pytest.approx(expected)
    assert scored["prediction"].tolist() == (expected >= 0.4).astype(int).tolist()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda frame: frame.iloc[0:0], "at least one row"),
        (lambda frame: frame.drop(columns=["V1"]), "missing input columns"),
        (lambda frame: frame.assign(Class=0), "extra input columns"),
        (lambda frame: frame.assign(V1="bad"), "nonnumeric input columns"),
        (lambda frame: frame.assign(V1=np.inf), "non-finite"),
    ],
)
def test_score_transactions_rejects_invalid_raw_frames(mutate, message):
    bundle, raw = _bundle()
    with pytest.raises(ValueError, match=message):
        score_transactions(bundle, mutate(raw))


def _written_run(tmp_path):
    path = tmp_path / "data.csv"
    raw = pd.DataFrame(
        {
            "Time": np.arange(20, dtype=float),
            "Amount": np.resize([1.0, 100.0], 20),
            "V1": np.resize([0.0, 4.0], 20),
            "Class": np.resize([0, 1], 20),
        }
    )
    raw.to_csv(path, index=False)
    artifact_dir = tmp_path / "runs"
    main(
        [
            "--data-path",
            str(path),
            "--batch-size",
            "20",
            "--test-size",
            "0.2",
            "--model",
            "logistic-regression",
            "--artifact-dir",
            str(artifact_dir),
        ]
    )
    return next(artifact_dir.iterdir()), raw


def test_load_bundle_requires_explicit_trust(tmp_path):
    run_dir, _ = _written_run(tmp_path)
    with pytest.raises(
        ValueError,
        match="Refusing to load pickle-based artifact without trusted=True",
    ):
        load_bundle(run_dir)


def test_load_bundle_rejects_hash_mismatch_before_deserialization(tmp_path, monkeypatch):
    import fraud_detection.inference as inference

    run_dir, _ = _written_run(tmp_path)
    with (run_dir / "bundle.joblib").open("ab") as bundle_file:
        bundle_file.write(b"corrupt")
    load_called = False

    def fail_if_called(*args, **kwargs):
        nonlocal load_called
        load_called = True
        raise AssertionError("joblib.load must not run")

    monkeypatch.setattr(inference.joblib, "load", fail_if_called)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_bundle(run_dir, trusted=True)
    assert load_called is False


def test_load_bundle_rejects_config_schema_before_deserialization(tmp_path, monkeypatch):
    import fraud_detection.inference as inference

    run_dir, _ = _written_run(tmp_path)
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text())
    config["schema_version"] = 2
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(
        inference.joblib,
        "load",
        lambda *args, **kwargs: pytest.fail("joblib.load must not run"),
    )
    with pytest.raises(ValueError, match="unsupported config schema_version"):
        load_bundle(run_dir, trusted=True)


def test_train_artifact_load_score_round_trip_matches_evaluation(tmp_path):
    run_dir, raw = _written_run(tmp_path)
    bundle = load_bundle(run_dir, trusted=True)
    scored = score_transactions(bundle, raw.drop(columns=["Class"]).iloc[-4:])
    with np.load(run_dir / "evaluation.npz") as evaluation:
        expected_scores = evaluation["test_scores"]
        expected_labels = evaluation["test_labels"]

    assert scored["fraud_probability"].to_numpy() == pytest.approx(expected_scores)
    assert len(expected_labels) == len(scored)
