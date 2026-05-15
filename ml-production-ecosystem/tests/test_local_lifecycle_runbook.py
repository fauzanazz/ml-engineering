from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "smoke-local-lifecycle.sh"
DEPLOYMENT_SCRIPT_PATH = ROOT / "scripts" / "smoke-local-deployment.sh"
DOC_PATH = ROOT / "docs" / "local-lifecycle-runbook.md"


def test_local_lifecycle_smoke_script_exists_and_checks_outputs() -> None:
    assert SCRIPT_PATH.exists()
    assert os.access(SCRIPT_PATH, os.X_OK)
    script = SCRIPT_PATH.read_text()

    assert "production-apply-local-platform" in script
    assert "production-validate-local-secret-injections" in script
    assert "production-validate-local-kubernetes" in script
    assert "production-validate-local-scheduler" in script
    assert "production-lifecycle-demo" in script
    assert "production-lifecycle-status" in script
    assert "configs/local-lifecycle-demo.yaml" in script
    assert "local lifecycle smoke passed" in script
    assert "local-lifecycle-status.json" in script
    for section in ["platform", "model_contract", "dataset", "offline_validation"]:
        assert section in script


def test_local_lifecycle_runbook_documents_no_download_smoke() -> None:
    assert DOC_PATH.exists()
    doc = DOC_PATH.read_text()

    assert "./scripts/smoke-local-lifecycle.sh" in doc
    assert "without downloading MovieLens" in doc
    assert "production-validate-platform-plan" in doc
    assert "production-validate-model-contract" in doc
    assert "local-lifecycle-demo.json" in doc

def test_local_deployment_smoke_script_starts_api_and_checks_drift() -> None:
    assert DEPLOYMENT_SCRIPT_PATH.exists()
    assert os.access(DEPLOYMENT_SCRIPT_PATH, os.X_OK)
    script = DEPLOYMENT_SCRIPT_PATH.read_text()

    assert "production-lifecycle-demo" in script
    assert "--approve" in script
    assert "--set-active" in script
    assert "foundation-serve-recommender" in script
    assert "/predict/v1" in script
    assert "production-demo-deployment" in script
    assert "production-detect-drift" in script
    assert "production-canary-decision" in script
    assert "production-continual-decision" in script
    assert "production-continual-summary" in script
    assert "production-lifecycle-status" in script
    assert "local-deployment-status.json" in script
    assert "local deployment smoke passed" in script
