#!/usr/bin/env bash
set -euo pipefail

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
  tests/test_deployment_manifest.py \
  tests/test_release_summary.py
