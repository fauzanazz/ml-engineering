import json
from pathlib import Path

import pandas as pd

from house_prices.cli import main


def _write_data(tmp_path: Path) -> tuple[Path, Path]:
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    rows = []
    for row_id in range(1, 31):
        rows.append(
            {
                "Id": row_id,
                "LotArea": 7000 + row_id * 100,
                "OverallQual": 5 + row_id % 4,
                "GrLivArea": 1000 + row_id * 15,
                "TotalBsmtSF": 600 + row_id * 10,
                "1stFlrSF": 800 + row_id * 8,
                "2ndFlrSF": 100 + row_id * 3,
                "YrSold": 2006 + row_id % 5,
                "MoSold": 1 + row_id % 12,
                "YearBuilt": 1980 + row_id % 25,
                "Neighborhood": ["A", "B", "C"][row_id % 3],
                "SalePrice": 120000 + row_id * 5000,
            }
        )
    pd.DataFrame(rows).to_csv(train_path, index=False)
    pd.DataFrame([{key: value for key, value in rows[0].items() if key != "SalePrice"}]).to_csv(test_path, index=False)
    return train_path, test_path


def test_cli_writes_artifacts_and_best_submission(tmp_path: Path):
    train_path, test_path = _write_data(tmp_path)
    artifacts_dir = tmp_path / "artifacts"

    status = main([
        "--data-path", str(train_path),
        "--test-path", str(test_path),
        "--artifacts-dir", str(artifacts_dir),
        "--model", "random-forest",
        "--run-id", "smoke",
    ])

    assert status == 0
    run_dir = artifacts_dir / "smoke"
    assert (run_dir / "config.json").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "model.joblib").exists()
    assert (run_dir / "feature_pipeline.joblib").exists()
    assert (run_dir / "submission.csv").exists()
    assert (artifacts_dir / "best_submission.csv").exists()
    config = json.loads((run_dir / "config.json").read_text())
    assert config["selection_metric"] == "validation_log_rmse"
    assert "leaderboard" in config["leakage_policy"]
