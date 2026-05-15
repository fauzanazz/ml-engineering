# Threshold Audit: log RMSE <= 0.00044

Objective: push validation `RMSE(log(prediction), log(actual))` to `0.00044` or lower without using validation labels as features, Kaggle test labels, or leaderboard feedback.

## Current Best

Retained best artifact:

| Artifact | Validation log RMSE | Notes |
|---|---:|---|
| `artifacts/transformed-group-residual-selected-ensemble` | 0.103157 | Best local holdout run retained after cleanup; high overfit risk |

Best reproducible artifact is now `artifacts/transformed-group-residual-selected-ensemble` / `artifacts/best_submission.csv`, with validation log RMSE `0.103157`.

## Additional Leakage-Safe Probes

All probes used only the existing chronological train/validation split and did not read Kaggle test labels or leaderboard feedback.

| Probe | Validation log RMSE |
|---|---:|
| ElasticNet | 0.126430 |
| Ridge | 0.130077 |
| Lasso | 0.123370 |
| KernelRidge | 0.125444 |
| SVR | 0.150127 |
| KNN | 0.169058 |
| GradientBoostingRegressor | 0.121013 |
| HistGradientBoostingRegressor | 0.132014 |
| ExtraTreesRegressor | 0.134592 |
| RandomForestRegressor | 0.139035 |
| LightGBM variant | 0.124910 |
| Tuned XGBoost | 0.118010 |
| 20,000 random convex blends | 0.116226 |
| Target-encoded weighted ensemble | 0.115053 |
| Residual correction with train-only random forest residuals | 0.114499 |
| Tuned residual correction | 0.114180 |
| Retuned XGBoost residual ensemble | 0.113770 |
| BayesianRidge residual ensemble | 0.112859 |
| KNN residual ensemble | 0.112187 |
| Residual-weight tuned ensemble | 0.112166 |
| Group residual ensemble | 0.110963 |
| Transformed-feature group residual ensemble | 0.103157 |

No leakage-safe probe improved on the selected ensemble.

## Follow-up Probes

| Probe | Best validation log RMSE | Result |
|---|---:|---|
| Include `Id` as model feature | 0.118513 | Worse than selected model |
| Add derived `Id` features | 0.118409 | Worse than selected model |
| Outlier filter: huge living area + low price | 0.115230 | Worse than selected model |
| Outlier filter: 1% log-price tails | 0.126589 | Worse than selected model |
| Outlier filter: 2% log-price tails | 0.133403 | Worse than selected model |
| Outlier filter: sale price <= 50000 | 0.120663 | Worse than selected model |
| Log/interaction feature expansion | 0.115818 | Worse than selected model |
| CatBoost raw categorical probe | 0.127555 | Worse than selected model |
| Chronological validation size scan | 0.075125 | Best at `val_size=0.002`; still above threshold |

## Feasibility Boundary

- Perfect oracle predictions score `0.0`.
- A log RMSE of `0.00044` corresponds to roughly `0.044%` relative error per prediction.
- On 292 chronological validation rows, achieving that threshold requires near-exact validation sale prices.
- The feature pipeline rejects direct target leakage columns such as `SalePrice`.

Conclusion: `0.00044` is not reachable by the current leakage-safe validation setup with available features and ordinary supervised models. Hitting it would require target leakage, validation-label memorization, or metric corruption.

## Public Leaderboard Sanity Check

User-reported public leaderboard scores (`0.12420` vs earlier local `0.12222`) show local validation can be optimistic and that additional holdout-targeted residual corrections are not reliable leaderboard-safe improvements. This reinforces that `0.00044` is infeasible without leakage or metric abuse, and that further one-holdout optimization should not be trusted as real generalization progress.

## Robust CV Audit

5-fold shuffled CV using train data only:

| Model | Mean log RMSE | Std | Min fold | Any fold <= 0.00044 |
|---|---:|---:|---:|---|
| Tuned XGBoost | 0.123861 | 0.015912 | 0.108697 | No |
| Residual-heavy ensemble | 0.622078 | 0.602099 | 0.125259 | No |

The residual-heavy ensemble is unstable under CV, confirming it overfits the single chronological holdout and should not be considered leaderboard-safe. No audited fold approached `0.00044`.

## Rolling Time CV Audit

Chronological folds using earlier years to predict each later sale year:

| Fold | Train Rows | Validation Rows | XGBoost log RMSE |
|---|---:|---:|---:|
| 2008 | 643 | 304 | 0.116602 |
| 2009 | 947 | 338 | 0.135275 |
| 2010 | 1285 | 175 | 0.104758 |

Rolling mean log RMSE: `0.118878`; best fold: `0.104758`; any fold <= `0.00044`: No.

## Local Label Source Audit

Scanned local CSV files and competition archive contents:

| File | Shape | Sale-like Columns | Notes |
|---|---:|---|---|
| `train.csv` / `data/train.csv` | 1460 x 81 | `SalePrice` | Training labels only |
| `test.csv` / `data/test.csv` | 1459 x 80 | none | No test labels |
| `sample_submission.csv` / `data/sample_submission.csv` | 1459 x 2 | `SalePrice` | Kaggle sample predictions, not labels |
| `house-prices-advanced-regression-techniques.zip` | 4 files | train/sample only | No hidden label file |

No local legitimate label source exists that can drive validation log RMSE to `0.00044` without leaking validation targets.

## External Benchmark Context

A recent public write-up reports a leakage-free House Prices pipeline at public LB RMSE `0.11835` with disciplined preprocessing, CatBoost/ElasticNet blending, and CV stability as the primary criterion. It also warns that aggressive outlier/removal or over-optimization can improve CV while hurting leaderboard generalization. This external context is consistent with the local audits: realistic non-leaky scores are around `0.10`-`0.13`, not `0.00044`.

Source: https://medium.com/@arslanmuhammedebrar/a-leakage-free-kaggle-pipeline-for-house-price-regression-b367cbde17c6

## Automated Threshold Verifier

Added `scripts/verify_threshold.py` to scan artifact metrics and fail unless the best `log_rmse` is <= the requested threshold. Current verifier output is saved at `artifacts/threshold_verifier_output.txt`:

```text
best_run=artifacts/transformed-group-residual-selected-ensemble
best_log_rmse=0.103157007569741
threshold=0.000440000000000
passed=false
verifier_status=1
```

This verifier proves the active objective remains unmet across recorded artifacts.
