"""Provider swap matrix tests."""

from pathlib import Path
import json
import subprocess
import sys

from ml_production_ecosystem.production_patterns.provider_swap_matrix import build_provider_swap_matrix

ROOT = Path(__file__).resolve().parents[1]


def test_provider_swap_matrix_documents_all_provider_resources(tmp_path: Path) -> None:
    report = build_provider_swap_matrix(ROOT, "development", tmp_path / "swap.json")

    assert report["status"] == "passed"
    assert report["core_code_changes_required"] is False
    assert report["providers"] == ["local", "aws", "gcp", "azure"]
    assert report["resource_matrix"]["model-artifacts"]["local"]["kind"] == "local-path"
    assert report["resource_matrix"]["model-artifacts"]["aws"]["uri"].startswith("s3://")
    assert report["resource_matrix"]["model-artifacts"]["gcp"]["uri"].startswith("gs://")
    assert report["resource_matrix"]["model-artifacts"]["azure"]["uri"].startswith("azblob://")
    assert "production-lifecycle-demo" in report["core_workflows"]
    assert "production-rollback-model" in report["core_workflows"]
    assert report["missing_resources_by_provider"]["aws"] == []
    assert "model-serving" in report["missing_resources_by_provider"]["local"]
    assert json.loads((tmp_path / "swap.json").read_text()) == report


def test_provider_swap_matrix_cli_writes_report(tmp_path: Path) -> None:
    output_path = tmp_path / "swap.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ml_production_ecosystem.production_patterns.provider_swap_matrix",
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
    assert "core_code_changes_required" in result.stdout
