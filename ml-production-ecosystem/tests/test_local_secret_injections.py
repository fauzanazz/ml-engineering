"""Local secret injection manifest validation tests."""

from pathlib import Path
import json
import shutil
import subprocess
import sys

from ml_production_ecosystem.production_patterns.local_secret_injections import validate_local_secret_injections

ROOT = Path(__file__).resolve().parents[1]


def _copy_platform(tmp_path: Path) -> None:
    shutil.copytree(ROOT / "04-platform-and-cloud", tmp_path / "04-platform-and-cloud")


def test_local_secret_injections_match_local_plan(tmp_path: Path) -> None:
    report = validate_local_secret_injections(ROOT, tmp_path / "local-secrets.json")

    assert report["status"] == "passed"
    assert report["required_targets"] == ["LOCAL_MODEL_REGISTRY_TOKEN"]
    assert report["declared_targets"] == ["LOCAL_MODEL_REGISTRY_TOKEN"]
    assert report["failures"] == []
    assert json.loads((tmp_path / "local-secrets.json").read_text()) == report


def test_local_secret_injections_fail_when_target_missing(tmp_path: Path) -> None:
    _copy_platform(tmp_path)
    manifest_path = tmp_path / "04-platform-and-cloud" / "secrets" / "local" / "secret-injections.yaml"
    manifest_path.write_text(manifest_path.read_text().replace("LOCAL_MODEL_REGISTRY_TOKEN", "OTHER_TOKEN"))

    report = validate_local_secret_injections(tmp_path, tmp_path / "local-secrets.json")

    assert report["status"] == "failed"
    assert "missing local injection target: LOCAL_MODEL_REGISTRY_TOKEN" in report["failures"]


def test_local_secret_injections_fail_when_values_not_external_only(tmp_path: Path) -> None:
    _copy_platform(tmp_path)
    manifest_path = tmp_path / "04-platform-and-cloud" / "secrets" / "local" / "secret-injections.yaml"
    manifest_path.write_text(manifest_path.read_text().replace("external-only", "inline"))

    report = validate_local_secret_injections(tmp_path, tmp_path / "local-secrets.json")

    assert report["status"] == "failed"
    assert "injections[0] value_handling must be external-only" in report["failures"]


def test_local_secret_injections_cli_writes_report(tmp_path: Path) -> None:
    output_path = tmp_path / "local-secrets.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ml_production_ecosystem.production_patterns.local_secret_injections",
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
    assert "LOCAL_MODEL_REGISTRY_TOKEN" in result.stdout
