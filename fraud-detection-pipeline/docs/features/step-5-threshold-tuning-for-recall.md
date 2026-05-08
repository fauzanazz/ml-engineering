# Step 5: Threshold Tuning for Recall

## Goal

Fraud detection favours recall over precision — a missed fraud (false negative) costs more than a false alarm (false positive). Default threshold 0.5 is arbitrary; lower thresholds catch more fraud at the cost of more false positives. This step adds configurable threshold control and a sweep table.

## Prior Steps

- [Step 1: Project Scaffold and One-Batch Baseline](step-1-project-scaffold-and-one-batch-baseline.md)
- [Step 2: Training Evaluation](step-2-training-evaluation.md)
- [Step 4: Run Logging and Artifacts](step-4-run-logging-and-artifacts.md)

## What Changed

### New: `src/fraud_detection/thresholds.py`

Pure functions — no side effects, no state.

| Symbol | Purpose |
|---|---|
| `validate_threshold(t)` | Raises `ValueError` if `t` not in `(0, 1)` |
| `apply_threshold(scores, threshold)` | Converts probability array to binary labels |
| `ThresholdRow` | Frozen dataclass: threshold, precision, recall, f1, FP, FN |
| `sweep_thresholds(labels, scores)` | Evaluates `[0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95]`, returns `list[ThresholdRow]` |

### Updated: `training.py`

- `train_one_batch` accepts `decision_threshold: float = 0.5`
- Uses `apply_threshold` instead of raw `model.predict` for test labels
- `TrainingResult` gains `test_labels` and `test_scores` fields (needed for sweep)

### Updated: `cli.py`

- `--decision-threshold` — float in `(0, 1)`, default `0.5`; validated via `ArgumentTypeError`
- `--threshold-sweep` — flag; prints precision/recall/f1/FP/FN table across 9 thresholds (0.05–0.95)
- `config` dict written to `config.json` includes `decision_threshold`

## CLI Usage

```bash
# default threshold
fraud-detect-train --data-path data/creditcard.csv --batch-size 50000 \
  --imbalance-strategy scale-pos-weight --no-artifacts

# lower threshold to catch more fraud
fraud-detect-train --data-path data/creditcard.csv --batch-size 50000 \
  --imbalance-strategy scale-pos-weight --decision-threshold 0.2 --no-artifacts

# sweep table
fraud-detect-train --data-path data/creditcard.csv --batch-size 50000 \
  --imbalance-strategy scale-pos-weight --threshold-sweep --no-artifacts
```

### Example sweep output

```
 threshold  precision     recall       f1     FP     FN
      0.05     ...        ...        ...     ...    ...
      0.10     ...        ...        ...     ...    ...
      0.20     ...        ...        ...     ...    ...
      0.30     ...        ...        ...     ...    ...
      0.50     ...        ...        ...     ...    ...
      0.70     ...        ...        ...     ...    ...
      0.80     ...        ...        ...     ...    ...
      0.90     ...        ...        ...     ...    ...
      0.95     ...        ...        ...     ...    ...
```

The sweep covers both directions from 0.5. Lower thresholds (0.05–0.30) trade precision for recall — fewer missed frauds, more false alarms. Upper thresholds (0.70–0.95) trade recall for precision — only high-confidence fraud flagged, fewer false positives but more misses. Run with real data to populate actual values.

## Tests Added

| File | Tests |
|---|---|
| `tests/test_thresholds.py` | validate rejects bad values; apply_threshold correctness; lower threshold reduces FN; sweep returns 5 rows with required fields |
| `tests/test_threshold_cli.py` | default threshold=0.5; custom threshold; rejects 0/1/out-of-range; sweep flag defaults false; sweep flag accepted |
| `tests/test_threshold_artifacts.py` | config.json includes decision_threshold |

71 tests total, all passing.
