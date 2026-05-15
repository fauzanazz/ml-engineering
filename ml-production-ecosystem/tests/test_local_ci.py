from pathlib import Path
import os

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
SCRIPT_PATH = ROOT / "scripts" / "validate-production-patterns.sh"
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "ml-production-ecosystem-ci.yml"
DOC_PATH = ROOT / "02-production-patterns" / "docs" / "local-ci.md"
REQUIRED_TESTS = [
    "tests/test_production_patterns_scaffold.py",
    "tests/test_production_retraining.py",
    "tests/test_quality_gate.py",
    "tests/test_monitoring_loop.py",
    "tests/test_production_monitoring_loop.py",
    "tests/test_scheduled_retraining.py",
    "tests/test_airflow_retraining_dag.py",
    "tests/test_data_ingestion.py",
    "tests/test_offline_validation.py",
    "tests/test_approval.py",
    "tests/test_deployment_demo.py",
    "tests/test_canary_decision.py",
    "tests/test_canary_router.py",
    "tests/test_drift_detection.py",
    "tests/test_continual_learning.py",
    "tests/test_continual_summary.py",
    "tests/test_lifecycle_graph.py",
    "tests/test_lifecycle_demo.py",
    "tests/test_lifecycle_status.py",
    "tests/test_goal_readiness.py",
    "tests/test_local_lifecycle_demo_config.py",
    "tests/test_local_lifecycle_runbook.py",
    "tests/test_generic_classifier_command.py",
    "tests/test_model_contract_manifest.py",
    "tests/test_recommendation_prediction_adapter.py",
    "tests/test_platform_plan.py",
    "tests/test_local_platform.py",
    "tests/test_platform_plan_adapter.py",
    "tests/test_cloud_provider_adapters.py",
    "tests/test_provider_boundaries.py",
    "tests/test_provider_portability.py",
    "tests/test_provider_swap_matrix.py",
    "tests/test_secret_references.py",
    "tests/test_local_secret_injections.py",
    "tests/test_local_kubernetes.py",
    "tests/test_local_scheduler.py",
    "tests/test_policy_references.py",
    "tests/test_shared_contracts.py",
    "tests/test_alerting_rules.py",
    "tests/test_rollback.py",
    "tests/test_rollback_generic_classifier.py",
    "tests/test_release_checklist.py",
    "tests/test_deployment_manifest.py",
    "tests/test_release_summary.py",
    "tests/test_scope_review.py",
    "tests/test_slo_burn_rate.py",
    "tests/test_burn_rate_alert.py",
    "tests/test_autoscaling_decision.py",
    "tests/test_load_aggregate.py",
    "tests/test_cost_estimate.py",
]


def test_validate_production_patterns_script_exists_executable_and_targets_tests() -> None:
    assert SCRIPT_PATH.exists()
    assert os.access(SCRIPT_PATH, os.X_OK)
    script = SCRIPT_PATH.read_text()

    assert script.startswith("#!/usr/bin/env bash")
    assert "uv run pytest" in script
    for test_path in REQUIRED_TESTS:
        assert test_path in script


def test_github_actions_workflow_runs_production_patterns_validation() -> None:
    assert WORKFLOW_PATH.exists()
    workflow = WORKFLOW_PATH.read_text()
    parsed_workflow = yaml.safe_load(workflow)

    assert parsed_workflow[True]["push"]
    assert parsed_workflow[True]["pull_request"]
    assert "ml-production-ecosystem/**" in workflow
    assert ".github/workflows/ml-production-ecosystem-ci.yml" in workflow
    assert "actions/checkout" in workflow
    assert "python-version: '3.13'" in workflow
    assert "astral-sh/setup-uv" in workflow
    assert "cd ml-production-ecosystem" in workflow
    assert "./scripts/validate-production-patterns.sh" in workflow


def test_local_ci_doc_explains_when_to_run_script() -> None:
    assert DOC_PATH.exists()
    doc = DOC_PATH.read_text()

    assert "./scripts/validate-production-patterns.sh" in doc
    assert "before push" in doc
    assert "before release" in doc
    assert "GitHub Actions" in doc
    assert "remote CI" in doc
    assert "production patterns" in doc
    for test_path in REQUIRED_TESTS:
        assert test_path in doc
