# Step 7: Hyperparameter Tuning

## Goal

Step 7 adds adaptive hyperparameter tuning for tree-based fraud models while preserving validation/test separation.

## Prior Steps

- [Step 6: Feature Engineering & Validation Split](step-6-feature-engineering-validation-split.md)
- [Artifact Security](../artifact-security.md)

## What Changed

### Updated: `src/fraud_detection/tuning.py`

New tuning module wraps Optuna TPE (Tree-structured Parzen Estimator) for supported model families:

| Function | Model | Default scoring | CV |
|---|---|---|---|
| `tune_lightgbm(...)` | LightGBM | `roc_auc` | `StratifiedKFold` |
| `tune_random_forest(...)` | Random Forest | `roc_auc` | `StratifiedKFold` |
| `tune_xgboost(...)` | XGBoost | `roc_auc` | `StratifiedKFold` |

Each tuning function returns a `TuningResult`:

| Field | Purpose |
|---|---|
| `best_params` | Best hyperparameters found by Optuna TPE |
| `best_score` | Best cross-validation score |
| `scoring` | Scoring metric used for search |
| `best_factory` | `ModelFactory`-compatible tuned model factory |

Input validation rejects unsafe tuning setups before search runs:

- `n_iter < 1`
  - `n_iter > 500` through CLI validation
- single-class targets
- minority class with fewer than 2 rows
- explicit `cv < 2`
- explicit `cv` greater than minority class count

### Updated: `src/fraud_detection/cli.py`

New tuning flags:

| Flag | Default | Purpose |
|---|---:|---|
| `--tune` | `False` | Run adaptive hyperparameter search before final training |
| `--tune-n-candidates` | `10` | Number of Optuna TPE trials; capped at 500. Each trial evaluates one hyperparameter configuration via full-data StratifiedKFold CV. |
| `--tune-n-iter` | — | Legacy alias for `--tune-n-candidates`; kept for backward compatibility, suppressed from help. |

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
  --tune-n-candidates 1 \
  --no-artifacts
```

Latest 500-candidate Optuna tuning comparison:

Commands:

```bash
uv run fraud-detect-train --data-path data/creditcard.csv --batch-size 10000 \
  --model lightgbm --tune --tune-n-candidates 500 --no-artifacts

uv run fraud-detect-train --data-path data/creditcard.csv --batch-size 10000 \
  --model random-forest --tune --tune-n-candidates 500 --no-artifacts

uv run fraud-detect-train --data-path data/creditcard.csv --batch-size 10000 \
  --model xgboost --tune --tune-n-candidates 500 --no-artifacts
```

| Model | Tuning CV ROC AUC | Precision | Recall | F1 | PR AUC | ROC AUC | Single-row latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Random Forest** | 0.9990 | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | 0.003094500s |
| LightGBM | 0.9997 | 0.8667 | **1.0000** | 0.9286 | 0.9897 | 0.9999 | **0.000319167s** |
| XGBoost | **0.9998** | 0.8571 | 0.9231 | 0.8889 | 0.9757 | 0.9998 | 0.000534916s |

Best held-out test model for this `batch_size=10000` experiment is **Random Forest**:

```text
tuning_best_params={'n_estimators': 200, 'max_depth': 5, 'min_samples_split': 2, 'min_samples_leaf': 4, 'max_features': 'sqrt'}
training_accuracy=0.9996
test_accuracy=1.0000
precision=1.0000
recall=1.0000
f1=1.0000
pr_auc=1.0000 roc_auc=1.0000
single_row_latency_s=0.003094500
```

LightGBM remains the best latency model with perfect recall and very high ROC AUC:

```text
tuning_best_params={'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.01, 'num_leaves': 63, 'subsample': 1.0}
precision=0.8667
recall=1.0000
f1=0.9286
pr_auc=0.9897 roc_auc=0.9999
single_row_latency_s=0.000319167
```

No single-class fold warnings — Optuna TPE runs full-data StratifiedKFold CV on every trial. The perfect Random Forest score is useful for this small research batch, but should be validated on a larger batch/full dataset before treating it as production evidence.

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
  --tune-n-candidates 20
```

## Decisions & Trade-offs

**Optuna TPE over HalvingRandomSearchCV.** HalvingRandomSearchCV subsamples training data across resource rounds, which creates single-class CV folds on imbalanced sorted fraud data and triggers ROC AUC warnings. Optuna TPE evaluates every trial with full-data StratifiedKFold CV, eliminating the subsampling problem entirely. TPE also builds a probabilistic model of the objective surface and concentrates samples in promising regions, which random search cannot do.

**ROC AUC as tuning objective.** ROC AUC is threshold-independent and complements PR AUC in evaluation. Fraud is imbalanced, so PR AUC remains important for final interpretation.

**Stratified CV over plain integer CV.** Explicit `StratifiedKFold` keeps class balance in folds and avoids single-class validation folds when class counts make CV feasible.

**Validation/test separation preserved.** Hyperparameters are selected from train-only cross-validation. Validation remains available for threshold selection; test remains final evaluation only.

**Lazy model imports.** LightGBM tuning does not import XGBoost, and XGBoost tuning does not import LightGBM. Model-specific dependencies fail only on model-specific paths.

**Optuna logs suppressed.** `optuna.logging.set_verbosity(logging.WARNING)` silences per-trial INFO logs. Warnings and errors remain visible.

## Known Limitations

- Search spaces are small, hand-written categorical distributions.
- `--tune-n-candidates` can still be expensive near the 500 cap.
- Tuning optimizes `roc_auc`; a production fraud policy may prefer `average_precision`, recall at precision floor, or cost-weighted utility.
- Tuning currently prints best score/params but does not persist tuning metadata into artifacts as a dedicated tuning report.

## Tests Added

307 tests pass with:

```bash
uv run pytest -q
```

Key coverage:

| File | Coverage |
|---|---|
| `tests/test_tuning.py` | Tuning result shape, best factory creation, CV validation, random state preservation, lazy imports, LightGBM/RF/XGBoost tuning, Optuna TPE sampler usage, full-data CV, no single-class warnings |
| `tests/test_cli_tuning.py` | CLI `--tune`, `--tune-n-candidates` validation, legacy `--tune-n-iter` alias, LightGBM tuning path, validation-split tuning path |
| `tests/test_metrics.py` | ROC AUC computation and single-class handling |
| `tests/test_training.py` | Single-row latency measurement and no-`predict_proba` fallback |

## Related

- [Step 6: Feature Engineering & Validation Split](step-6-feature-engineering-validation-split.md)
- [Run Log](../run-log.md)
- [Artifact Security](../artifact-security.md)
