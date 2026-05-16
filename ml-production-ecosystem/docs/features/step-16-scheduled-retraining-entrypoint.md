# Step 16: Scheduled Retraining Entrypoint

## Goal

Add a scheduler-friendly retraining command that runs config-driven retraining and emits a machine-readable run summary.

## User Story

Sebagai ML engineer, gw bisa run satu command untuk scheduled retraining flow: train model baru, apply quality gate, optionally activate model, lalu dapat status selesai/gagal dalam JSON.

## Command

```bash
uv run production-scheduled-retrain \
  --config configs/foundation-recommender.yaml \
  --set-active \
  --require-quality-gate \
  --output-path artifacts/reports/production-patterns/scheduled-retraining.json
```

## What It Reuses

This step reuses [production retraining](../../src/ml_production_ecosystem/production_patterns/retraining.py) instead of duplicating training logic. The scheduled entrypoint wraps `production_patterns.retraining.run_retraining` so future schedulers can call one stable command.

## Output

Successful run:

```json
{
  "status": "completed",
  "config_path": "configs/foundation-recommender.yaml",
  "set_active": true,
  "require_quality_gate": true,
  "run_id": "foundation-config-v1",
  "model_name": "recommendation",
  "quality_gate": {
    "status": "passed"
  }
}
```

Failure output:

```json
{
  "status": "failed",
  "config_path": "configs/foundation-recommender.yaml",
  "error": "quality gate failed"
}
```

When `--output-path` is provided, the same summary is written to a JSON report for scheduler logs, release evidence, and later runbook references.

## Key Files

- `src/ml_production_ecosystem/production_patterns/scheduled_retraining.py`
- `src/ml_production_ecosystem/production_patterns/retraining.py`
- `pyproject.toml` script: `production-scheduled-retrain`
- `tests/test_scheduled_retraining.py`
- `docs/domains/production-patterns/retraining.md`
- `docs/domains/production-patterns/release-checklist.md`

## Pattern

```text
scheduler/terminal
  -> production-scheduled-retrain
  -> run_retraining
  -> train candidate
  -> quality gate
  -> optional activation
  -> JSON summary/report
```

## Out Of Scope

- Real Airflow deployment.
- Cron daemon.
- Docker scheduler service.
- Retry/backoff logic.
- Notifications.
- Remote artifact store.
- Multi-model orchestration.

## Acceptance Criteria

- `production-scheduled-retrain --config ...` returns JSON summary.
- Successful run writes optional report when `--output-path` is provided.
- Quality gate failure returns `status = "failed"`, not stack trace.
- Command reuses existing retraining code, no duplicate training logic.
- Tests mock retraining path or use small fixture config.
- Existing tests stay green.

## Definition Of Done

`production-patterns domain` has scheduler-ready retraining entrypoint. Project covers train → gate → serve → observe → monitor → scheduled retrain summary.

## Next Step

[Step 17](./step-17-airflow-dag-skeleton.md) adds an Airflow DAG skeleton that calls this CLI.
