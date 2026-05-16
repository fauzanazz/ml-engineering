# Step 22: Lightweight CI Validation

## Goal

Add a local CI validation command/script that runs core checks for `production-patterns domain` without needing an external CI provider.

## User Story

Sebagai ML engineer, gw bisa run satu command sebelum push untuk validasi production-patterns: tests, manifest parse, alert rules parse, checklist presence, dan DAG import safety.

## Command

```bash
./scripts/validate-production-patterns.sh
```

The script runs focused tests:

```bash
uv run pytest \
  tests/test_production_patterns_scaffold.py \
  tests/test_production_retraining.py \
  tests/test_quality_gate.py \
  tests/test_monitoring_loop.py \
  tests/test_production_monitoring_loop.py \
  tests/test_scheduled_retraining.py \
  tests/test_airflow_retraining_dag.py \
  tests/test_alerting_rules.py \
  tests/test_rollback.py \
  tests/test_release_checklist.py \
  tests/test_deployment_manifest.py
```

The current script also includes release summary coverage because [Step 26](./step-26-release-summary-report.md) later added `tests/test_release_summary.py` to the focused validation set.

## Documentation

`docs/domains/production-patterns/local-ci.md` explains when to run local CI before push or before release.

## Key Files

- `scripts/validate-production-patterns.sh`
- `docs/domains/production-patterns/local-ci.md`
- `tests/test_local_ci.py`

## Pattern

```text
local change
  -> ./scripts/validate-production-patterns.sh
  -> focused production-pattern tests
  -> fast confidence before push/release
```

## Out Of Scope

- GitHub Actions.
- Pre-commit hook.
- Docker build validation.
- Full integration test with live API.
- Coverage reporting.
- Lint/format gate.

## Acceptance Criteria

- Script exists and executable.
- Script runs targeted `02` tests.
- Doc explains when to run it before push/release.
- Tests assert script references required test files.
- Existing tests stay green.

## Definition Of Done

`production-patterns domain` has repeatable local validation loop. Project covers train → gate → activate → serve → monitor → alert → rollback → deploy metadata → local CI.

## Next Step

[Step 23](./step-23-github-actions-ci-workflow.md) adds GitHub Actions remote CI skeleton.
