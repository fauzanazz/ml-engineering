"""Local Kubernetes manifest validation tests."""

from pathlib import Path
import json
import subprocess
import sys

from ml_production_ecosystem.production_patterns.local_kubernetes import validate_local_kubernetes

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "platform" / "local" / "kubernetes" / "foundation-api.yaml"


def test_local_kubernetes_manifest_passes(tmp_path: Path) -> None:
    report = validate_local_kubernetes(MANIFEST, tmp_path / "k8s.json")

    assert report["status"] == "passed"
    assert report["kinds"] == ["ConfigMap", "Deployment", "Service"]
    assert report["failures"] == []
    assert json.loads((tmp_path / "k8s.json").read_text()) == report


def test_local_kubernetes_rejects_committed_secret(tmp_path: Path) -> None:
    manifest = tmp_path / "bad.yaml"
    manifest.write_text(MANIFEST.read_text() + "\n---\napiVersion: v1\nkind: Secret\nmetadata:\n  name: bad\n")

    report = validate_local_kubernetes(manifest, tmp_path / "k8s.json")

    assert report["status"] == "failed"
    assert "forbidden committed kinds: Secret" in report["failures"]


def test_local_kubernetes_rejects_literal_env_secret(tmp_path: Path) -> None:
    manifest = tmp_path / "bad.yaml"
    manifest.write_text(MANIFEST.read_text().replace("valueFrom:\n                secretKeyRef:", "value: bad-secret\n              valueFrom:\n                secretKeyRef:"))

    report = validate_local_kubernetes(manifest, tmp_path / "k8s.json")

    assert report["status"] == "failed"
    assert "deployment env must not contain literal secret values" in report["failures"]


def test_local_kubernetes_cli_writes_report(tmp_path: Path) -> None:
    output = tmp_path / "k8s.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ml_production_ecosystem.production_patterns.local_kubernetes",
            "--manifest-path",
            str(MANIFEST),
            "--output-path",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    report = json.loads(output.read_text())
    assert report["status"] == "passed"
    assert "Deployment" in result.stdout
