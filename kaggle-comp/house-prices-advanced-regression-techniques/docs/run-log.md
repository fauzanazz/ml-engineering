# Experiment Run Log

Validation uses training data only. No Kaggle test labels or public leaderboard feedback are used for model choice.

## Split Policy

House Prices includes `YrSold` and `MoSold`, so validation is chronological: earlier sales train, latest sales validate. `Id` breaks ties only. If those columns are absent, code falls back to a seeded random holdout and records `random_holdout_no_reliable_chronology_columns` in `config.json`.

## Current Best Artifact

Command used for the retained best run:

```bash
uv run house-prices-train --data-path data/train.csv --test-path data/test.csv --artifacts-dir artifacts --model all --run-id transformed-group-residual-selected
```

| Retained Run | Model | Validation log RMSE | Notes |
|---|---|---:|---|
| `artifacts/transformed-group-residual-selected-ensemble` | Transformed-feature group residual ensemble | 0.103157 | Best local holdout run; likely overfit to validation and not leaderboard-safe |

Recommended local submission: `artifacts/best_submission.csv`, copied from `artifacts/transformed-group-residual-selected-ensemble/submission.csv`.

## Historical Experiment Summary

Transient run directories were removed during cleanup. Key validation milestones were:

| Stage | Validation log RMSE | Notes |
|---|---:|---|
| Baseline LightGBM | 0.133478 | Initial boosted-tree baseline |
| Baseline Random Forest | 0.139471 | Bagging baseline |
| Tuned XGBoost | 0.118010 | Optuna search on train-only chronological validation |
| Weighted ensemble | 0.116053 | XGBoost + linear blend |
| Target-encoded ensemble | 0.115053 | Smoothed train-only target means for selected categoricals |
| Residual-corrected ensemble | 0.114180 | Random-forest residual corrector |
| BayesianRidge residual ensemble | 0.112859 | Added BayesianRidge blend member |
| KNN residual ensemble | 0.112187 | Added 15-neighbor residual correction |
| Residual-weight tuned ensemble | 0.112166 | Tuned RF/KNN residual weights |
| Group residual ensemble | 0.110963 | Group mean corrections on raw feature groups |
| Transformed-feature group residual ensemble | 0.103157 | Best local holdout run; overfit risk high |

## Feature Pipeline

- Drops `Id` from model features while preserving it for submissions.
- Blocks target-like feature columns such as `SalePrice` before fitting or transforming.
- Handles missing numeric values with train-fitted medians.
- Handles categorical values with train-fitted most-frequent imputation, unknown-safe one-hot encoding, and smoothed train-only target means for selected high-signal columns.
- Adds House Prices features: total square footage, total bathrooms, house age at sale, remodel age, garage age, quality-area interaction, and quality score aggregates.
- Trains models on `log(SalePrice)` and evaluates `RMSE(log(prediction), log(actual))`.

## Selected Model

- Components: tuned XGBoost, ElasticNet, Lasso, GradientBoostingRegressor, BayesianRidge, plus random-forest, KNN, and grouped residual correctors fit on training residuals only.
- Chosen by local chronological validation only; no Kaggle test labels were used.
- Public leaderboard sanity check suggests the residual-heavy best local model is overfit and should not be treated as robust generalization.

## Next Steps

- Prefer robust CV-selected models over further single-holdout residual corrections.
- Keep Kaggle leaderboard feedback out of model selection.
