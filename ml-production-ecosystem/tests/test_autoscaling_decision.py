"""Autoscaling decision simulation tests."""

from pathlib import Path
import json
import subprocess
import sys

import pytest

from scale_reliability.autoscaling_decision import build_autoscaling_decision


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def test_autoscaling_decision_scales_up_on_slo_breach(tmp_path: Path) -> None:
    load = _write_json(
        tmp_path / "load.json",
        {"request_count": 120, "concurrency": 3, "error_count": 2, "latency_ms": {"p95": 250.0}},
    )
    slo = _write_json(tmp_path / "slo.json", {"status": "breached", "breached_slos": ["latency_p95_ms"]})

    report = build_autoscaling_decision(load, slo, current_replicas=1, max_replicas=4, output_path=tmp_path / "scale.json")

    assert report["action"] == "scale_up"
    assert report["desired_replicas"] == 3
    assert "latency_p95_ms" in report["breached_slos"]
    assert json.loads((tmp_path / "scale.json").read_text()) == report


def test_autoscaling_decision_holds_at_max_replicas(tmp_path: Path) -> None:
    load = _write_json(
        tmp_path / "load.json",
        {"request_count": 500, "concurrency": 10, "error_count": 5, "latency_ms": {"p95": 400.0}},
    )
    slo = _write_json(tmp_path / "slo.json", {"status": "breached", "breached_slos": ["error_rate"]})

    report = build_autoscaling_decision(load, slo, current_replicas=4, max_replicas=4, output_path=tmp_path / "scale.json")

    assert report["action"] == "hold"
    assert report["desired_replicas"] == 4


def test_autoscaling_decision_scales_down_when_underused(tmp_path: Path) -> None:
    load = _write_json(
        tmp_path / "load.json",
        {"request_count": 20, "concurrency": 1, "error_count": 0, "latency_ms": {"p95": 20.0}},
    )
    slo = _write_json(tmp_path / "slo.json", {"status": "within_budget", "breached_slos": []})

    report = build_autoscaling_decision(load, slo, current_replicas=3, min_replicas=1, max_replicas=4, output_path=tmp_path / "scale.json")

    assert report["action"] == "scale_down"
    assert report["desired_replicas"] == 2


def test_autoscaling_decision_rejects_invalid_bounds(tmp_path: Path) -> None:
    load = _write_json(tmp_path / "load.json", {"request_count": 1, "concurrency": 1, "error_count": 0})

    with pytest.raises(ValueError, match="current_replicas must be within min/max bounds"):
        build_autoscaling_decision(load, None, current_replicas=0, output_path=tmp_path / "scale.json")


def test_autoscaling_decision_cli_writes_report(tmp_path: Path) -> None:
    load = _write_json(
        tmp_path / "load.json",
        {"request_count": 60, "concurrency": 2, "error_count": 0, "latency_ms": {"p95": 30.0}},
    )
    output = tmp_path / "scale.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scale_reliability.autoscaling_decision",
            "--load-report",
            str(load),
            "--output-path",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    report = json.loads(output.read_text())
    assert report["action"] == "scale_up"
    assert "desired_replicas" in result.stdout
