# Step 19: Model Rollback Entrypoint

## Goal

Add a safe rollback command that switches active model version back to a known-good registered model.

## User Story

Sebagai ML engineer, gw bisa rollback active model via satu command ketika alerting atau monitoring nunjukin model baru bermasalah.

## Command

```bash
uv run production-rollback-model \
  --registry-path registry/models.json \
  --model-name movielens-popularity \
  --target-version foundation-config-v1 \
  --reason "high drift after deploy"
```

## Behavior

The rollback command:

1. Reads the local registry.
2. Validates the model exists.
3. Validates the target version exists.
4. Sets the target version as the active model.
5. Writes a rollback event report to `artifacts/reports/production-patterns/rollback.json`.
6. Prints a JSON summary.

## Output

Successful rollback:

```json
{
  "status": "rolled_back",
  "model_name": "movielens-popularity",
  "target_version": "foundation-config-v1",
  "reason": "high drift after deploy",
  "report_path": "artifacts/reports/production-patterns/rollback.json"
}
```

Failure output:

```json
{
  "status": "failed",
  "model_name": "movielens-popularity",
  "target_version": "missing-version",
  "error": "target version not found"
}
```

Failures return JSON instead of stack trace noise, so runbooks and future automation can consume the result safely.

## Key Files

- `src/ml_production_ecosystem/production_patterns/rollback.py`
- `pyproject.toml` script: `production-rollback-model`
- `artifacts/reports/production-patterns/rollback.json`
- `tests/test_rollback.py`
- `docs/domains/production-patterns/release-checklist.md`
- `docs/domains/production-patterns/alerting-runbook.md`

## Pattern

```text
alert or failed release verification
  -> choose known-good version
  -> production-rollback-model
  -> local registry active pointer update
  -> rollback report
  -> rerun production-monitor
```

## Out Of Scope

- Auto rollback from alerts.
- Canary deploy.
- Traffic splitting.
- Model binary restore from remote store.
- Registry database.
- Approval workflow.

## Acceptance Criteria

- `production-rollback-model` returns JSON summary.
- Invalid model/version returns `status = "failed"`, no stack trace.
- Successful rollback updates active model in local registry.
- Rollback report includes reason and target version.
- Tests use temp registry fixture.
- Existing tests stay green.

## Definition Of Done

`production-patterns domain` can recover from bad activation manually. Project covers train → gate → serve → observe → monitor → alert → rollback.

## Next Step

[Step 20](./step-20-deployment-release-checklist.md) adds a manual model release checklist.
