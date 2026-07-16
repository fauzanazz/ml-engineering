# Step 4: Run Logging & Artifacts

This step originally introduced the following legacy, step-specific format:

| File | Legacy contents |
|------|----------|
| `metrics.json` | Flat training/test metrics |
| `config.json` | Flat data, batch, split, imbalance, and model keys |
| `model.txt` | LightGBM-only booster text |

Current runs use the schema-version-1 `bundle.joblib`, `evaluation.npz`, nested JSON metadata, and SHA-256 manifests documented in the [README artifact contract](../../README.md#artifact-contract-and-trust). The table above is retained only as historical implementation context.

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
