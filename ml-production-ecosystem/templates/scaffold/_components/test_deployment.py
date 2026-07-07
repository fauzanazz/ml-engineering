from pathlib import Path

from {{package_name}}.deployment import run_local_deployment


def test_local_deployment_writes_report(tmp_path: Path) -> None:
    report_path = tmp_path / "artifacts" / "reports" / "deployment.json"

    report = run_local_deployment(output_path=report_path)

    assert report["status"] == "completed"
    assert report["provider"] == "{{provider}}"
    assert report_path.exists()
