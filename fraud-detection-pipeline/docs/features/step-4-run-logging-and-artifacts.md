# Step 4: Run Logging & Artifacts

Each `fraud-detect-train` run writes a timestamped folder under `artifacts/runs/` (default) containing:

| File | Contents |
|------|----------|
| `metrics.json` | `training_accuracy`, `test_accuracy`, `precision`, `recall`, `f1`, `pr_auc` |
| `config.json` | `data_path`, `batch_size`, `test_size`, `imbalance_strategy`, `model_name` |
| `model.txt` | LightGBM booster text format via `booster_.save_model()` |

## CLI flags

```
--artifact-dir PATH   base dir for run folders (default: artifacts/runs)
--no-artifacts        skip writing artifacts entirely
```

Run with custom dir:

```bash
fraud-detect-train --data-path data/creditcard.csv --batch-size 10000 \
  --artifact-dir /tmp/myexps
```

Run without artifacts:

```bash
fraud-detect-train --data-path data/creditcard.csv --no-artifacts
```

## Key files

- `src/fraud_detection/artifacts.py` — `write_artifacts()`, `make_run_dir()`
- `src/fraud_detection/training.py` — `TrainingResult.model` field added
- `src/fraud_detection/cli.py` — wires args → artifact write

## Design decisions

- `run_id` injectable for deterministic test paths; defaults to UTC timestamp
- `model` field on `TrainingResult` uses `compare=False` (frozen dataclass, model not equality-comparable)
- No predictions persisted — metrics + model sufficient for reproducibility
