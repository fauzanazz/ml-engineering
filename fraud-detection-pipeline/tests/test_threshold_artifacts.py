import json

import numpy as np
import pandas as pd
import pytest

from fraud_detection.cli import main
from fraud_detection.inference import load_bundle


def test_selected_threshold_matches_config_and_bundle(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "Time": np.arange(20, dtype=float),
            "Amount": np.resize([1.0, 100.0], 20),
            "V1": np.resize([0.0, 4.0], 20),
            "Class": np.resize([0, 1], 20),
        }
    ).to_csv(path, index=False)
    artifact_dir = tmp_path / "runs"

    main(
        [
            "--data-path",
            str(path),
            "--batch-size",
            "20",
            "--test-size",
            "0.2",
            "--val-size",
            "0.2",
            "--threshold-objective",
            "target-recall",
            "--target-recall",
            "0.95",
            "--model",
            "logistic-regression",
            "--artifact-dir",
            str(artifact_dir),
        ]
    )

    run_dir = next(artifact_dir.iterdir())
    config = json.loads((run_dir / "config.json").read_text())
    bundle = load_bundle(run_dir, trusted=True)
    assert bundle.effective_threshold == pytest.approx(
        config["threshold"]["selected"]
    )
