from pathlib import Path
import json

from ml_production_ecosystem.production_patterns.provider_boundaries import validate_provider_boundaries

def test_validate_provider_boundaries_allows_adapter_imports(tmp_path: Path) -> None:
    adapter_path = tmp_path / "04-platform-and-cloud" / "adapters" / "aws" / "adapter.py"
    core_path = tmp_path / "02-production-patterns" / "workflow.py"
    adapter_path.parent.mkdir(parents=True)
    core_path.parent.mkdir(parents=True)
    adapter_path.write_text("import boto3\n")
    core_path.write_text("from ml_production_ecosystem.shared.lifecycle import LifecycleRun\n")

    report = validate_provider_boundaries(tmp_path, tmp_path / "report.json")

    assert report["status"] == "passed"
    assert report["violations"] == []

def test_validate_provider_boundaries_rejects_core_vendor_imports(tmp_path: Path) -> None:
    source_path = tmp_path / "02-production-patterns" / "workflow.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("from google.cloud import storage\n")

    report = validate_provider_boundaries(tmp_path, tmp_path / "report.json")

    assert report["status"] == "failed"
    assert report["violations"] == ["02-production-patterns/workflow.py"]

def test_validate_provider_boundaries_writes_report(tmp_path: Path) -> None:
    output_path = tmp_path / "reports" / "provider-boundaries.json"

    report = validate_provider_boundaries(tmp_path, output_path)

    assert json.loads(output_path.read_text()) == report
