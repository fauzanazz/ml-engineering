"""Local platform apply command tests."""

from pathlib import Path
import json
import subprocess
import sys

from ml_production_ecosystem.production_patterns.local_platform import apply_local_platform

ROOT = Path(__file__).resolve().parents[1]


def _copy_adapter(tmp_path: Path) -> Path:
    source = ROOT / "04-platform-and-cloud" / "adapters" / "local" / "adapter.py"
    target = tmp_path / "04-platform-and-cloud" / "adapters" / "local" / "adapter.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source.read_text())
    return target


def test_apply_local_platform_creates_filesystem_resources(tmp_path: Path) -> None:
    _copy_adapter(tmp_path)
    output_path = tmp_path / "reports" / "local-platform.json"

    summary = apply_local_platform(tmp_path, "development", output_path)

    assert summary["status"] == "ready"
    assert summary["provider"] == "local"
    assert (tmp_path / "01-foundation" / "artifacts").is_dir()
    assert (tmp_path / "01-foundation" / "logs").is_dir()
    assert (tmp_path / "01-foundation" / "registry").is_dir()
    assert "LOCAL_MODEL_REGISTRY_TOKEN" in summary["secret_references"]
    assert json.loads(output_path.read_text()) == summary


def test_apply_local_platform_cli_writes_report(tmp_path: Path) -> None:
    _copy_adapter(tmp_path)
    output_path = tmp_path / "reports" / "local-platform.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ml_production_ecosystem.production_patterns.local_platform",
            "--project-root",
            str(tmp_path),
            "--output-path",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(output_path.read_text())
    assert payload["status"] == "ready"
    assert "model-artifacts" in result.stdout
