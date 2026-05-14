from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "validate-production-patterns.sh"
DOC_PATH = ROOT / "02-production-patterns" / "docs" / "local-ci.md"
REQUIRED_TESTS = [
    "tests/test_production_patterns_scaffold.py",
    "tests/test_production_retraining.py",
    "tests/test_quality_gate.py",
    "tests/test_monitoring_loop.py",
    "tests/test_production_monitoring_loop.py",
    "tests/test_scheduled_retraining.py",
    "tests/test_airflow_retraining_dag.py",
    "tests/test_alerting_rules.py",
    "tests/test_rollback.py",
    "tests/test_release_checklist.py",
    "tests/test_deployment_manifest.py",
]


def test_validate_production_patterns_script_exists_executable_and_targets_tests() -> None:
    assert SCRIPT_PATH.exists()
    assert os.access(SCRIPT_PATH, os.X_OK)
    script = SCRIPT_PATH.read_text()

    assert script.startswith("#!/usr/bin/env bash")
    assert "uv run pytest" in script
    for test_path in REQUIRED_TESTS:
        assert test_path in script


def test_local_ci_doc_explains_when_to_run_script() -> None:
    assert DOC_PATH.exists()
    doc = DOC_PATH.read_text()

    assert "./scripts/validate-production-patterns.sh" in doc
    assert "before push" in doc
    assert "before release" in doc
    assert "production patterns" in doc
    for test_path in REQUIRED_TESTS:
        assert test_path in doc
