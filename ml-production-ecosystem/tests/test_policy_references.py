"""Policy-as-code reference validation tests."""

from pathlib import Path
import json
import shutil
import subprocess
import sys

from ml_production_ecosystem.production_patterns.policy_references import validate_policy_references

ROOT = Path(__file__).resolve().parents[1]


def test_policy_references_pass_current_platform_plans(tmp_path: Path) -> None:
    report = validate_policy_references(ROOT, tmp_path / "policies.json")

    assert report["status"] == "passed"
    assert report["policy_count"] == 4
    assert report["reference_count"] == 4
    assert "local-env-file-reference" in report["policy_refs"]
    assert "policy/aws-model-registry-read" in report["policy_refs"]
    assert report["failures"] == []
    assert json.loads((tmp_path / "policies.json").read_text()) == report


def test_policy_references_fail_missing_policy(tmp_path: Path) -> None:
    shutil.copytree(ROOT / "configs" / "platform", tmp_path / "configs" / "platform")
    (tmp_path / "configs" / "platform" / "policies" / "aws" / "model-registry-read.yaml").unlink()

    report = validate_policy_references(tmp_path, tmp_path / "policies.json")

    assert report["status"] == "failed"
    assert "missing policy_ref: policy/aws-model-registry-read" in report["failures"]


def test_policy_references_fail_injection_mismatch(tmp_path: Path) -> None:
    shutil.copytree(ROOT / "configs" / "platform", tmp_path / "configs" / "platform")
    policy_path = tmp_path / "configs" / "platform" / "policies" / "gcp" / "model-registry-read.yaml"
    policy_path.write_text(policy_path.read_text().replace("MODEL_REGISTRY_TOKEN", "GCP_ONLY_TOKEN"))

    report = validate_policy_references(tmp_path, tmp_path / "policies.json")

    assert report["status"] == "failed"
    assert "injection target mismatch for policy/gcp-model-registry-read" in report["failures"]


def test_policy_references_cli_writes_report(tmp_path: Path) -> None:
    output_path = tmp_path / "policies.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ml_production_ecosystem.production_patterns.policy_references",
            "--root",
            str(ROOT),
            "--output-path",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    report = json.loads(output_path.read_text())
    assert report["status"] == "passed"
    assert "policy/aws-model-registry-read" in result.stdout
