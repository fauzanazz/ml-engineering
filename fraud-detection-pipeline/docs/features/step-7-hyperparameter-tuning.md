# Step 7: Hyperparameter Tuning

## Goal

Step 7 adds randomized hyperparameter tuning for tree-based fraud models while preserving validation/test separation.

## Prior Steps

- [Step 6: Feature Engineering & Validation Split](step-6-feature-engineering-validation-split.md)
- [Artifact Security](../artifact-security.md)

## What Changed

### Updated: `src/fraud_detection/tuning.py`

New tuning module wraps `RandomizedSearchCV` for supported model families:

| Function | Model | Default scoring | CV |
|---|---|---|---|
| `tune_lightgbm(...)` | LightGBM | `roc_auc` | `StratifiedKFold` |
| `tune_random_forest(...)` | Random Forest | `roc_auc` | `StratifiedKFold` |
| `tune_xgboost(...)` | XGBoost | `roc_auc` | `StratifiedKFold` |

Each tuning function returns a `TuningResult`:

| Field | Purpose |
|---|---|
| `best_params` | Best hyperparameters found by randomized search |
| `best_score` | Best cross-validation score |
| `scoring` | Scoring metric used for search |
| `best_factory` | `ModelFactory`-compatible tuned model factory |

Input validation rejects unsafe tuning setups before cross-validation runs:

- `n_iter < 1`
- `n_iter > 200` through CLI validation
- single-class targets
- minority class with fewer than 2 rows
- explicit `cv < 2`
- explicit `cv` greater than minority class count

### Updated: `src/fraud_detection/cli.py`

New tuning flags:

| Flag | Default | Purpose |
|---|---:|---|
| `--tune` | `False` | Run randomized hyperparameter search before final training |
| `--tune-n-iter` | `10` | Number of sampled parameter combinations; capped at 200 |

Supported tuning models:

```text
lightgbm
random-forest
xgboost
```

When `--val-size` is provided, tuning uses the same chronological train split produced by `load_three_way_split()`. Validation and test rows stay out of hyperparameter search.

### Updated Metrics

Step 7 also records broader evaluation/serving signals added alongside tuning:

- `roc_auc` next to `pr_auc`
- `val_roc_auc` when validation metrics exist
- `single_row_latency_s` for online-style one-row `predict_proba` latency

## CLI Usage

Tune LightGBM:

```bash
uv run fraud-detect-train \
  --data-path data/creditcard.csv \
  --batch-size 10000 \
  --model lightgbm \
  --tune \
  --tune-n-iter 1 \
  --no-artifacts
```

Latest 100-iteration LightGBM tuning result:

```bash
uv run fraud-detect-train --data-path data/creditcard.csv --batch-size 10000 \
  --model lightgbm --tune --tune-n-iter 100 --no-artifacts
```

```text
tuning_best_score=0.9997
tuning_best_params={'subsample': 0.6, 'num_leaves': 127, 'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.01}
training_accuracy=0.9999
test_accuracy=0.9990
precision=0.8667
recall=1.0000
f1=0.9286
pr_auc=0.9897 roc_auc=0.9999
predict_proba_latency_s=0.002295
predict_proba_latency_per_row_s=0.000001149
single_row_latency_s=0.000331958
```

This is the current best documented LightGBM tuning run for the `batch_size=10000` smoke setup: it reaches perfect recall (`1.0000`) with very high ROC AUC (`0.9999`) and improved F1 (`0.9286`). The tuning objective was train-only cross-validation ROC AUC, while test metrics above remain final held-out evaluation.

Use a larger search for real runs:

```bash
uv run fraud-detect-train \
  --data-path data/creditcard.csv \
  --batch-size 50000 \
  --imbalance-strategy scale-pos-weight \
  --val-size 0.1 \
  --threshold-objective target-recall \
  --target-recall 0.95 \
  --model random-forest \
  --tune \
  --tune-n-iter 20
```

## Decisions & Trade-offs

**RandomizedSearchCV over a heavier AutoML dependency.** The project already depends on scikit-learn, so randomized search adds useful tuning without introducing Optuna, Ray, or another orchestration layer.

**ROC AUC as tuning objective.** ROC AUC is threshold-independent and complements PR AUC in evaluation. Fraud is imbalanced, so PR AUC remains important for final interpretation.

**Stratified CV over plain integer CV.** Explicit `StratifiedKFold` keeps class balance in folds and avoids single-class validation folds when class counts make CV feasible.

**Validation/test separation preserved.** Hyperparameters are selected from train-only cross-validation. Validation remains available for threshold selection; test remains final evaluation only.

**Lazy model imports.** LightGBM tuning does not import XGBoost, and XGBoost tuning does not import LightGBM. Model-specific dependencies fail only on model-specific paths.

## Known Limitations

- Search spaces are small, hand-written distributions.
- `--tune-n-iter` can still be expensive near the 200 cap.
- Tuning optimizes `roc_auc`; a production fraud policy may prefer `average_precision`, recall at precision floor, or cost-weighted utility.
- Tuning currently prints best score/params but does not persist tuning metadata into artifacts as a dedicated tuning report.

## Tests Added

294 tests pass with:

```bash
uv run pytest -q
```

Key coverage:

| File | Coverage |
|---|---|
| `tests/test_tuning.py` | Tuning result shape, best factory creation, CV validation, random state preservation, lazy imports, LightGBM/RF/XGBoost tuning |
| `tests/test_cli_tuning.py` | CLI `--tune`, `--tune-n-iter` validation, LightGBM tuning path, validation-split tuning path |
| `tests/test_metrics.py` | ROC AUC computation and single-class handling |
| `tests/test_training.py` | Single-row latency measurement and no-`predict_proba` fallback |

## Related

- [Step 6: Feature Engineering & Validation Split](step-6-feature-engineering-validation-split.md)
- [Run Log](../run-log.md)
- [Artifact Security](../artifact-security.md)
