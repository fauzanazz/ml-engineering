"""Local cost estimate tests."""

from pathlib import Path
import json
import subprocess
import sys

import pytest

from scale_reliability.cost_estimate import build_cost_estimate


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def test_cost_estimate_uses_autoscaling_and_load_reports(tmp_path: Path) -> None:
    autoscaling = _write_json(
        tmp_path / "autoscaling.json",
        {"current_replicas": 1, "desired_replicas": 3, "load": {"request_count": 50}},
    )
    load = _write_json(tmp_path / "load.json", {"request_count": 100})

    report = build_cost_estimate(
        autoscaling,
        load,
        tmp_path / "cost.json",
        replica_hourly_cost=0.25,
        hours_per_month=100,
    )

    assert report["status"] == "estimated"
    assert report["delta_replicas"] == 2
    assert report["estimated_hourly_cost"] == 0.75
    assert report["estimated_monthly_cost"] == 75.0
    assert report["estimated_cost_per_1000_requests"] == 7.5
    assert json.loads((tmp_path / "cost.json").read_text()) == report


def test_cost_estimate_falls_back_to_autoscaling_load_basis(tmp_path: Path) -> None:
    autoscaling = _write_json(
        tmp_path / "autoscaling.json",
        {"current_replicas": 2, "desired_replicas": 2, "load": {"request_count": 200}},
    )

    report = build_cost_estimate(autoscaling, None, tmp_path / "cost.json", replica_hourly_cost=0.1)

    assert report["request_count_basis"] == 200
    assert report["estimated_cost_per_1000_requests"] == 1.0


def test_cost_estimate_rejects_invalid_inputs(tmp_path: Path) -> None:
    autoscaling = _write_json(tmp_path / "autoscaling.json", {"current_replicas": 1, "desired_replicas": 1})

    with pytest.raises(ValueError, match="replica_hourly_cost must be non-negative"):
        build_cost_estimate(autoscaling, None, tmp_path / "cost.json", replica_hourly_cost=-1)

    with pytest.raises(ValueError, match="hours_per_month must be at least 1"):
        build_cost_estimate(autoscaling, None, tmp_path / "cost.json", hours_per_month=0)


def test_cost_estimate_cli_writes_report(tmp_path: Path) -> None:
    autoscaling = _write_json(tmp_path / "autoscaling.json", {"current_replicas": 1, "desired_replicas": 2})
    output = tmp_path / "cost.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scale_reliability.cost_estimate",
            "--autoscaling-report",
            str(autoscaling),
            "--output-path",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    report = json.loads(output.read_text())
    assert report["desired_replicas"] == 2
    assert "estimated_monthly_cost" in result.stdout
