# Step 6: Feature Engineering & Validation Split

## Goal

Step 5 threshold tuning used test data to pick the operating threshold — test-peeking invalidates held-out metrics. Step 6 fixes this by introducing a true three-way chronological split (train / val / test), fitting a `FeaturePipeline` on train only, and tuning threshold on val only. Test remains untouched until final evaluation.

## Prior Steps

- [Step 4: Run Logging and Artifacts](step-4-run-logging-and-artifacts.md)
- [Step 5: Threshold Tuning for Recall](step-5-threshold-tuning-for-recall.md) — exploratory; threshold selected on test set (now superseded for production use)

## What Changed

### New: `src/fraud_detection/features.py`

Stateful `FeaturePipeline` class; fit on train, transform all splits.

| Symbol | Purpose |
|---|---|
| `FeaturePipeline.fit(df)` | Learns `RobustScaler` from train features |
| `FeaturePipeline.transform(df)` | Applies engineering + fitted scaler |
| `_engineer(df, v_cols)` | Pure function: adds `log_amount_raw`, `amount_is_zero`, `hour_of_day`, `day`, `is_night` |
| `_SCALE_COLS` | `["log_amount_raw", "Amount"]` — inputs to scaler |
| `_SCALE_OUTPUT` | `["log_amount_scaled", "amount_scaled"]` — scaled outputs |

Engineered features:

| Feature | Derivation |
|---|---|
| `log_amount_raw` | `log1p(Amount)` |
| `amount_is_zero` | `1` if `Amount == 0` else `0` |
| `hour_of_day` | `Time % 86400 // 3600` |
| `day` | `Time // 86400` |
| `is_night` | `1` if `hour_of_day < 6` |
| `log_amount_scaled` | `log_amount_raw` after `RobustScaler` |
| `amount_scaled` | `Amount` after `RobustScaler` |

`Amount` raw is dropped from final output (replaced by `amount_scaled`).

### Updated: `src/fraud_detection/data.py`

New `load_three_way_split()` function. Splits sorted-by-`Time` rows into `[train | val | test]` using chronological cutpoints, then:

1. Deduplicates each split independently (`drop_duplicates` on feature cols).
2. Removes val rows whose feature hash exists in train.
3. Removes test rows whose feature hash exists in train or val.

This prevents cross-split leakage without shuffling the time order. The existing `load_time_split_batch` received the same dedup + overlap-removal treatment in Step 6.

### Updated: `src/fraud_detection/thresholds.py`

New `select_threshold_on_validation()` function selects best threshold from a val sweep:

| Objective | Selection rule | Fallback |
|---|---|---|
| `f1` | max F1, tie-break: higher threshold | — |
| `target-recall` | among `recall >= target_recall`: max precision, then recall, then threshold | max F1 row if none qualify |

### Updated: `src/fraud_detection/training.py`

`train_one_batch` dispatches to `_train_with_validation()` when `val_size` is provided. New fields on `TrainingResult`:

| Field | Type | Content |
|---|---|---|
| `val_threshold` | `float \| None` | Threshold chosen on val |
| `val_metrics` | `ClassificationMetrics \| None` | Metrics at val threshold on val set |
| `threshold_objective` | `str \| None` | `"f1"` or `"target-recall"` |
| `target_recall` | `float \| None` | Floor used with `target-recall` objective |
| `split_counts` | `SplitCounts \| None` | Row counts for train/val/test |

### Updated: `src/fraud_detection/artifacts.py`

`write_artifacts` conditionally appends to `metrics.json`:

- `val_threshold`, `val_precision`, `val_recall`, `val_f1`, `val_pr_auc` — when `val_metrics` present
- `split_train`, `split_val`, `split_test` — when `split_counts` present

### Updated: `src/fraud_detection/cli.py`

New args:

| Flag | Default | Purpose |
|---|---|---|
| `--val-size FLOAT` | `None` | Enables three-way split |
| `--threshold-objective {f1,target-recall}` | `f1` | Objective for val threshold selection |
| `--target-recall FLOAT` | `None` | Recall floor for `target-recall` objective |

Validation rules enforced at parse time:
- `--target-recall` requires `--threshold-objective target-recall`
- `--threshold-objective target-recall` requires `--val-size`
- `--target-recall` requires `--val-size`

## CLI Usage

```bash
# Two-way split (original behaviour, unchanged)
uv run fraud-detect-train --data-path data/creditcard.csv --batch-size 50000 \
  --imbalance-strategy scale-pos-weight

# Three-way split, pick threshold by best F1 on val
uv run fraud-detect-train --data-path data/creditcard.csv --batch-size 50000 \
  --imbalance-strategy scale-pos-weight --val-size 0.1

# Three-way split, pick threshold to hit ≥0.90 recall on val
uv run fraud-detect-train --data-path data/creditcard.csv --batch-size 50000 \
  --imbalance-strategy scale-pos-weight --val-size 0.1 \
  --threshold-objective target-recall --target-recall 0.90

# Skip artifact write
uv run fraud-detect-train --data-path data/creditcard.csv --batch-size 50000 \
  --val-size 0.1 --no-artifacts
```

## Decisions & Trade-offs

**Chronological split over random split.** Fraud patterns drift with time; a random split leaks future signal into train. Sorting by `Time` and cutting at index boundaries preserves the deployment-time assumption that train precedes inference.

**FeaturePipeline fit on train only.** Fitting the scaler on all data (including val/test) is a common source of leakage. `FeaturePipeline.fit` is called once on `raw_train_feat`; `transform` is applied to each split using the same fitted state.

**Hash-based overlap removal over row-index exclusion.** Duplicate rows with identical feature vectors across splits can cause a model trained on them to trivially "memorise" test answers. Hashing feature columns (via `pd.util.hash_pandas_object`) and filtering is fast and set-safe.

**Validation threshold, test evaluation.** `select_threshold_on_validation` only sees val labels. The chosen `val_threshold` is then applied to test scores — test set never participates in threshold selection.

**Step 5 threshold sweep retained as diagnostic.** `--threshold-sweep` still works in two-way mode and is useful for exploratory analysis. It is not a substitute for proper val-based tuning.

## Known Limitations

- `select_threshold_on_validation` sweeps only 9 candidate thresholds (`_DEFAULT_SWEEP`). A fine-grained sweep or precision-recall curve search would find a better operating point on large data.
- `target-recall` fallback to max-F1 row is silent — no warning emitted if target cannot be met.
- `load_three_way_split` reads the full `batch_size` rows then splits; for very large files this may load more data than strictly needed for the val/test fraction.

## Tests Added

164 tests total, all passing (`uv run pytest -q`).

| File | Coverage |
|---|---|
| `tests/test_features.py` | FeaturePipeline fit/transform correctness; leakage guard (scaler not re-fit on transform); schema validation; missing column errors |
| `tests/test_data.py` | Split proportions; chronological ordering; dedup; cross-split overlap removal |
| `tests/test_thresholds.py` | `select_threshold_on_validation` f1 and target-recall objectives; fallback behaviour |
| `tests/test_training.py` | `_train_with_validation` populates `val_threshold`, `val_metrics`, `split_counts` |
| `tests/test_artifacts.py` | `metrics.json` includes val metrics and split counts when present |
| `tests/test_cli.py` | `--val-size` accepted; arg validation errors for bad objective/recall combos |
| `tests/test_review_blockers.py` | Regression coverage for CLI validation, schema validation, artifact invariants, split counts |

## Related

- [Step 5: Threshold Tuning for Recall](step-5-threshold-tuning-for-recall.md)
- [`src/fraud_detection/features.py`](../../src/fraud_detection/features.py)
- [`src/fraud_detection/data.py`](../../src/fraud_detection/data.py)
- [`src/fraud_detection/thresholds.py`](../../src/fraud_detection/thresholds.py)
- [`src/fraud_detection/training.py`](../../src/fraud_detection/training.py)
