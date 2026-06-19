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

Run this before push when changing `production-patterns domain`, production CLIs, alert docs, deployment manifest metadata, or scheduler DAG skeleton.

Run this before push when changing local lifecycle contracts, local lifecycle demo config, model contract manifests, provider-neutral platform plan validation, or provider boundary enforcement.

Run this before release as a lightweight local gate after following [release-checklist.md](./release-checklist.md) and before relying on [deployment-manifest.md](./deployment-manifest.md) metadata.

For enterprise-grade evidence, run:

```bash
./scripts/validate-enterprise-readiness.sh
```

This script runs local production-pattern CI, full pytest (with optional RT warehouse services), enterprise control validations, smoke checks, monitoring, load/burn-rate/safety chain, scheduled retrain, and release-summary artifact generation where possible.

## What it checks

The script runs focused production patterns tests:

- `tests/test_production_patterns_scaffold.py`
- `tests/test_production_retraining.py`
- `tests/test_quality_gate.py`
- `tests/test_monitoring_loop.py`
- `tests/test_production_monitoring_loop.py`
- `tests/test_scheduled_retraining.py`
- `tests/test_airflow_retraining_dag.py`
- `tests/test_data_ingestion.py`
- `tests/test_offline_validation.py`
- `tests/test_approval.py`
- `tests/test_deployment_demo.py`
- `tests/test_canary_decision.py`
- `tests/test_canary_router.py`
- `tests/test_drift_detection.py`
- `tests/test_continual_learning.py`
- `tests/test_continual_summary.py`
- `tests/test_lifecycle_graph.py`
- `tests/test_lifecycle_demo.py`
- `tests/test_lifecycle_status.py`
- `tests/test_goal_readiness.py`
- `tests/test_local_lifecycle_demo_config.py`
- `tests/test_local_lifecycle_runbook.py`
- `tests/test_generic_classifier_command.py`
- `tests/test_model_contract_manifest.py`
- `tests/test_recommendation_prediction_adapter.py`
- `tests/test_platform_plan.py`
- `tests/test_local_platform.py`
- `tests/test_platform_apply.py`
- `tests/test_platform_plan_adapter.py`
- `tests/test_cloud_provider_adapters.py`
- `tests/test_provider_boundaries.py`
- `tests/test_provider_portability.py`
- `tests/test_provider_swap_matrix.py`
- `tests/test_secret_references.py`
- `tests/test_local_secret_injections.py`
- `tests/test_local_kubernetes.py`
- `tests/test_local_scheduler.py`
- `tests/test_policy_references.py`
- `tests/test_shared_contracts.py`
- `tests/test_alerting_rules.py`
- `tests/test_rollback.py`
- `tests/test_rollback_generic_classifier.py`
- `tests/test_release_checklist.py`
- `tests/test_deployment_manifest.py`
- `tests/test_release_summary.py`
- `tests/test_scope_review.py`
- `tests/test_slo_burn_rate.py`
- `tests/test_burn_rate_alert.py`
- `tests/test_autoscaling_decision.py`
- `tests/test_load_aggregate.py`
- `tests/test_cost_estimate.py`

This covers scaffold presence, retraining, quality gate, monitoring, legacy monitoring loop compatibility, scheduled retraining, DAG import safety, local lifecycle commands, local canary release decision, local traffic-splitting canary simulation, cross-provider platform apply proof, local lifecycle demo config and runbook, lifecycle status summary, goal readiness audit, continual-learning history summary, command-trained non-recommender sample model, model contracts, generic prediction adapter bridge, provider-neutral platform plan validation, platform plan adapter loading, cloud provider adapter loading, provider boundary enforcement, provider portability checks, provider swap matrix proof, secret-reference enforcement, local secret injection validation, local Kubernetes manifest validation, local scheduler plan validation and runtime dry-run, policies-as-code reference validation, shared contracts, alert rules parsing, rollback behavior including command-trained generic classifier rollback, release checklist presence, deployment manifest parsing, release summary generation, and scope review closure, and local SLO burn-rate simulation, multi-window burn-rate alert simulation, and autoscaling decision simulation, and distributed load report aggregation, and local cost estimation.

## Limits

This is a focused local CI command mirrored by GitHub Actions remote CI. It does not run Docker builds, start a live API, collect coverage, enforce formatting, or validate a production scheduler.
