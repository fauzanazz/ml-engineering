# Step 7 — Temporal Hyperparameter Tuning

## Current contract

Hyperparameter tuning is restricted to LightGBM, Random Forest, and XGBoost. Every public tuner has the same controls:

```python
tune_*(
    X,
    y,
    *,
    n_iter=10,
    cv=3,
    scoring="average_precision",
    random_state=42,
    imbalance_strategy="none",
)
```

Only `average_precision` is accepted. Fraud is severely imbalanced, so this objective measures ranking quality for the minority class more directly than ROC AUC.

Each Optuna TPE trial is evaluated with forward-only `TimeSeriesSplit` on chronological training data:

1. Require monotonic nondecreasing `Time`.
2. Generate train-before-validation folds.
3. Fit a fresh `FeaturePipeline` on each fold's training rows.
4. Transform that fold's training and validation rows with the fold-local pipeline.
5. Recompute `scale_pos_weight` from that fold's training labels when requested.
6. Fit a fresh seeded estimator and score validation probabilities with average precision.

A fold is invalid if its training or validation partition lacks either class. The tuner retries with one fewer split down to two and records the effective fold count. It fails clearly if no valid two-fold temporal evaluation exists.

`TuningResult` records:

- `best_params`
- `best_score`
- `scoring="average_precision"`
- `best_factory`
- `cv_strategy="TimeSeriesSplit"`
- effective `cv_splits`
- `random_state`

The tuned factory uses the same seed and imbalance policy as candidate evaluation and final training.

## CLI

| Flag | Default | Purpose |
|---|---:|---|
| `--tune` | `False` | Run Optuna TPE before final training |
| `--tune-n-candidates` | `10` | Number of trials, capped at 500 |
| `--tune-cv` | `3` | Requested forward temporal folds, range 2–10 |
| `--seed` | `42` | Optuna, candidate-estimator, and final-estimator seed |

Example:

```bash
uv run fraud-detect-train \
  --data-path data/creditcard.csv \
  --batch-size 284807 \
  --model random-forest \
  --imbalance-strategy scale-pos-weight \
  --val-size 0.1 \
  --test-size 0.2 \
  --threshold-objective target-recall \
  --target-recall 0.95 \
  --tune \
  --tune-n-candidates 50 \
  --tune-cv 3 \
  --seed 42
```

The schema-version-1 run config persists trial count, objective, CV strategy, effective folds, best score, and best parameters. Validation remains reserved for threshold selection; the held-out temporal test remains final evaluation only.

## Canonical full-dataset result

The predeclared 284,807-row Random Forest benchmark used 50 trials and three effective temporal folds. The full protocol and observed metrics are in [`../full-dataset-benchmark.md`](../full-dataset-benchmark.md). The benchmark was not selected by comparing multiple held-out test results.

## Historical experiment

The earlier 10,000-row, 500-candidate comparison used shuffled stratified ROC-AUC CV and reported unusually strong held-out metrics. It remains dated research evidence in [`../run-log.md`](../run-log.md), not the current tuning protocol or canonical benchmark. Its small-slice results must not be interpreted as current production evidence.

## Trade-offs

- Optuna TPE evaluates every trial on all requested temporal folds; large trial counts are expensive.
- Automatic fold reduction preserves a valid temporal evaluation but reduces the number of estimates; the effective count is always recorded.
- One temporal holdout does not replace repeated production backtesting or drift monitoring.
- The search spaces remain explicit, small categorical distributions for auditability.
