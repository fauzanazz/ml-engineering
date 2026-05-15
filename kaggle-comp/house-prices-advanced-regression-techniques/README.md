# House Prices Regression Pipeline

Reusable ML framework for Kaggle's House Prices - Advanced Regression Techniques competition.

## Goals

- Optimize validation log RMSE using training data only.
- Avoid Kaggle test labels and public leaderboard feedback during tuning.
- Use chronology-aware validation when sale date columns exist.
- Save each run's config, metrics, model, feature pipeline, and submission.

## Setup

```bash
uv sync
unzip house-prices-advanced-regression-techniques.zip -d data
```

## Train One Model

```bash
uv run house-prices-train --data-path data/train.csv --test-path data/test.csv --artifacts-dir artifacts --model lightgbm
```

## Compare Model Families

```bash
uv run house-prices-train --data-path data/train.csv --test-path data/test.csv --artifacts-dir artifacts --model all --run-id validation-selected
```

The CLI writes one run directory per model. Selection uses validation log RMSE only. When `data/test.csv` exists, each run gets `submission.csv`; the best validation run is copied to `artifacts/best_submission.csv`.

Current best local validation run: target-encoded weighted log-prediction ensemble, log RMSE `0.103157`. Details: [run log](docs/run-log.md).

## Project Layout

- `src/house_prices/data.py`: CSV loading and validation split policy.
- `src/house_prices/features.py`: leakage checks, feature engineering, imputation, encoding.
- `src/house_prices/models.py`: model factories for LightGBM, Random Forest, XGBoost, and the selected weighted ensemble.
- `src/house_prices/training.py`: log-target training, validation metrics, submission prediction.
- `src/house_prices/artifacts.py`: run artifact writer.
- `src/house_prices/cli.py`: training CLI.
- `tests/`: split, feature, metrics, artifacts, training, and CLI smoke tests.

## Validation Policy

If `YrSold` and `MoSold` exist, validation uses the latest chronological rows as holdout. If they do not exist, validation falls back to a seeded random holdout and records that reason in `config.json`.
