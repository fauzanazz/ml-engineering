import json

import numpy as np
import pandas as pd

from fraud_detection.cli import main


def test_run_artifact_contains_auditable_split_and_no_legacy_model_files(tmp_path):
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
            "--model",
            "lightgbm",
            "--artifact-dir",
            str(artifact_dir),
        ]
    )

    run_dir = next(artifact_dir.iterdir())
    config = json.loads((run_dir / "config.json").read_text())
    assert not (run_dir / "model.txt").exists()
    assert not (run_dir / "model.joblib").exists()
    assert not (run_dir / "predictions.csv").exists()
    assert config["split"]["audit"]["val"] is not None
    for partition in ("train", "val", "test"):
        audit = config["split"]["audit"][partition]
        assert audit["rows"] == audit["positives"] + audit["negatives"]
        assert audit["time_min"] <= audit["time_max"]
