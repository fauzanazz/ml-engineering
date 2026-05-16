# Step 17: Airflow DAG Skeleton

## Goal

Add an Airflow-style DAG skeleton that documents scheduled retraining orchestration without requiring a real Airflow runtime yet.

## User Story

Sebagai ML engineer, gw bisa lihat bagaimana `production-scheduled-retrain` nanti dijalankan oleh scheduler, dengan task order jelas dan config tetap reusable.

## DAG Shape

The DAG skeleton documents this task order:

```text
validate_config
  -> scheduled_retrain
  -> monitor_after_retrain
```

The file is import-safe without Airflow installed. When Airflow is unavailable, placeholder objects preserve the documented task names and dependency order for tests and readers.

## Commands In DAG

Scheduled retraining command:

```bash
uv run production-scheduled-retrain \
  --config configs/foundation-recommender.yaml \
  --set-active \
  --require-quality-gate \
  --output-path artifacts/reports/production-patterns/scheduled-retraining.json
```

Post-retrain monitor command:

```bash
uv run production-monitor \
  --base-url http://127.0.0.1:8000 \
  --max-error-count 0 \
  --max-drift-score 0.2 \
  --max-latency-ms-last 100
```

## Key Files

- `configs/production-patterns/airflow/retraining_dag.py`
- `src/ml_production_ecosystem/production_patterns/scheduled_retraining.py`
- `src/ml_production_ecosystem/production_patterns/monitoring_loop.py`
- `tests/test_airflow_retraining_dag.py`
- `docs/domains/production-patterns/retraining.md`
- `docs/domains/production-patterns/monitoring-loop.md`

## Pattern

```text
Airflow later
  -> validate reusable config
  -> run scheduled retraining CLI
  -> monitor serving API after activation
  -> expose scheduler orchestration shape without runtime dependency
```

## Out Of Scope

- Installing Airflow.
- Running scheduler daemon.
- Docker Compose Airflow stack.
- Remote executor.
- Backfill/catchup behavior.
- Retries/alerts/SLAs.

## Acceptance Criteria

- DAG skeleton imports without Airflow installed, or degrades with clear placeholder objects.
- Task order is documented: validate → retrain → monitor.
- Commands reuse Step 15 and Step 16 CLIs.
- Tests assert DAG file exists and contains expected tasks/commands.
- Existing tests stay green.

## Definition Of Done

`production-patterns domain` shows first scheduler orchestration layer. Project covers train → gate → serve → observe → monitor → scheduled retrain → DAG skeleton.

## Next Step

Step 18 can add alerting rules/runbook or Airflow Docker runtime. This project chose alerting rules and an operational runbook next.
