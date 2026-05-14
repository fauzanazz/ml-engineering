---
title: Local CI For Production Patterns
type: feature-note
created: 2026-05-14
status: completed
categories: [ml-production, ci, validation]
related:
  - ./release-checklist.md
  - ./deployment-manifest.md
  - ./alerting-runbook.md
---

# Local CI For Production Patterns

Run `./scripts/validate-production-patterns.sh` from `ml-production-ecosystem` to validate production patterns before push or before release. This local CI mirrors GitHub Actions remote CI for `ml-production-ecosystem` changes.

## Command

```bash
./scripts/validate-production-patterns.sh
```

## When to run

Run this before push when changing `02-production-patterns`, production CLIs, alert docs, deployment manifest metadata, or scheduler DAG skeleton.

Run this before release as a lightweight local gate after following [release-checklist.md](./release-checklist.md) and before relying on [deployment-manifest.md](./deployment-manifest.md) metadata.

## What it checks

The script runs focused production patterns tests:

- `tests/test_production_patterns_scaffold.py`
- `tests/test_production_retraining.py`
- `tests/test_quality_gate.py`
- `tests/test_monitoring_loop.py`
- `tests/test_production_monitoring_loop.py`
- `tests/test_scheduled_retraining.py`
- `tests/test_airflow_retraining_dag.py`
- `tests/test_alerting_rules.py`
- `tests/test_rollback.py`
- `tests/test_release_checklist.py`
- `tests/test_deployment_manifest.py`

This covers scaffold presence, retraining, quality gate, monitoring, legacy monitoring loop compatibility, scheduled retraining, DAG import safety, alert rules parsing, rollback behavior, release checklist presence, and deployment manifest parsing.

## Limits

This is a focused local CI command mirrored by GitHub Actions remote CI. It does not run Docker builds, start a live API, collect coverage, enforce formatting, or validate a production scheduler.
