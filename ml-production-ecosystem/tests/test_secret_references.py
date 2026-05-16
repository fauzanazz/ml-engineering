from pathlib import Path
import json

from ml_production_ecosystem.production_patterns.secret_references import validate_secret_references

def test_validate_secret_references_passes_reference_only_configs(tmp_path: Path) -> None:
    plan_path = tmp_path / "configs" / "platform" / "iac" / "local" / "platform-plan.yaml"
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text(
        """
provider: local
environment: development
secrets:
  - provider: local
    name: development/local/model-registry-token
    injection_target: MODEL_REGISTRY_TOKEN
    policy_ref: local-env-file-reference
""".strip()
    )

    report = validate_secret_references(tmp_path, tmp_path / "report.json")

    assert report["status"] == "passed"
    assert report["violations"] == []
    assert json.loads((tmp_path / "report.json").read_text()) == report

def test_validate_secret_references_rejects_forbidden_value_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "bad.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        """
runtime:
  credentials:
    api_key: do-not-commit
    nested:
      password: also-bad
""".strip()
    )

    report = validate_secret_references(tmp_path, tmp_path / "report.json")

    assert report["status"] == "failed"
    assert report["violations"] == [
        {
            "path": "configs/bad.yaml",
            "location": "runtime.credentials.api_key",
            "reason": "forbidden secret value key",
        },
        {
            "path": "configs/bad.yaml",
            "location": "runtime.credentials.nested.password",
            "reason": "forbidden secret value key",
        },
    ]
