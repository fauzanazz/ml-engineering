# Step 26: Release Summary Report

## Goal

Add a command that writes one release summary JSON by combining retraining report, deployment manifest, monitor summary, smoke test status, and rollback plan.

## User Story

Sebagai ML engineer, setelah release manual, gw bisa generate satu file ringkasan yang menjawab: model apa dirilis, quality gate status apa, service apa, smoke/monitor sehat tidak, dan rollback target apa.

## Command

```bash
uv run production-release-summary \
  --retraining-report artifacts/reports/production-patterns/scheduled-retraining.json \
  --deployment-manifest configs/production-patterns/deploy/deployment-manifest.yaml \
  --monitor-summary artifacts/reports/production-patterns/monitoring-loop.json \
  --smoke-status passed \
  --rollback-target foundation-config-v1 \
  --output-path artifacts/reports/production-patterns/release-summary.json
```

## Output

Ready summary:

```json
{
  "status": "ready",
  "model_name": "movielens-popularity",
  "run_id": "foundation-config-v1",
  "quality_gate": {"status": "passed"},
  "service_name": "foundation-api",
  "image": "ml-production-ecosystem-foundation-api",
  "smoke_status": "passed",
  "monitor_status": "healthy",
  "rollback_target": "foundation-config-v1"
}
```

## Status Rules

`status` is `ready` only when:

- quality gate status is `passed`
- smoke status is `passed`
- monitor status is `healthy`

Otherwise, `status` is `blocked`. Missing input files return a blocked summary with an `error` field instead of stack trace noise.

## Key Files

- `src/ml_production_ecosystem/production_patterns/release_summary.py`
- `pyproject.toml` script: `production-release-summary`
- `artifacts/reports/production-patterns/release-summary.json`
- `tests/test_release_summary.py`
- `docs/domains/production-patterns/release-checklist.md`
- `docs/domains/production-patterns/local-ci.md`

## Pattern

```text
scheduled retraining report
  + deployment manifest
  + monitor summary
  + smoke status
  + rollback target
  -> production-release-summary
  -> release-summary.json
```

## Out Of Scope

- Auto running retrain/smoke/monitor.
- Uploading report.
- HTML/PDF report.
- Approval workflow.
- Slack/email.
- CI artifact upload.

## Acceptance Criteria

- `production-release-summary` returns JSON summary.
- Writes output report when `--output-path` is provided.
- Blocked status when smoke, monitor, or quality gate fail.
- Missing file handled as blocked summary.
- Tests use temp JSON/YAML fixtures.
- Existing tests stay green.

## Definition Of Done

`production-patterns domain` has release evidence artifact. Project covers train → gate → activate → serve → smoke test → monitor → alert → rollback → deploy metadata → CI → release summary.

## Next Step

Step 27 can be `production-patterns domain` scope review / gap checklist before moving to next topic.
