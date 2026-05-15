"""SLO burn-rate simulation tests."""

from pathlib import Path
import json
import subprocess
import sys

from scale_reliability.slo_burn_rate import build_slo_burn_rate_report


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def test_slo_burn_rate_within_budget(tmp_path: Path) -> None:
    load = _write_json(
        tmp_path / "load.json",
        {
            "request_count": 100,
            "success_count": 100,
            "error_count": 0,
            "latency_ms": {"p95": 120.0},
        },
    )
    drift = _write_json(tmp_path / "drift.json", {"score": 0.1})

    report = build_slo_burn_rate_report(load, drift, tmp_path / "slo.json")

    assert report["status"] == "within_budget"
    assert report["breached_slos"] == []
    assert report["checks"]["availability"]["burn_rate"] == 0.0
    assert report["checks"]["latency_p95_ms"]["status"] == "within_budget"
    assert json.loads((tmp_path / "slo.json").read_text()) == report


def test_slo_burn_rate_breaches_error_budget_and_latency(tmp_path: Path) -> None:
    load = _write_json(
        tmp_path / "load.json",
        {
            "request_count": 100,
            "success_count": 97,
            "error_count": 3,
            "latency_ms": {"p95": 250.0},
        },
    )

    report = build_slo_burn_rate_report(load, None, tmp_path / "slo.json")

    assert report["status"] == "breached"
    assert "availability" in report["breached_slos"]
    assert "error_rate" in report["breached_slos"]
    assert "latency_p95_ms" in report["breached_slos"]
    assert report["checks"]["error_rate"]["burn_rate"] == 3.0


def test_slo_burn_rate_breaches_drift_threshold(tmp_path: Path) -> None:
    load = _write_json(
        tmp_path / "load.json",
        {
            "request_count": 10,
            "success_count": 10,
            "error_count": 0,
            "latency_ms": {"p95": 10.0},
        },
    )
    drift = _write_json(tmp_path / "drift.json", {"payload": {"drift_score": 0.4}})

    report = build_slo_burn_rate_report(load, drift, tmp_path / "slo.json")

    assert report["status"] == "breached"
    assert report["checks"]["drift_score"]["burn_rate"] == 2.0


def test_slo_burn_rate_cli_writes_report(tmp_path: Path) -> None:
    load = _write_json(
        tmp_path / "load.json",
        {
            "request_count": 1,
            "success_count": 1,
            "error_count": 0,
            "latency_ms": {"p95": 5.0},
        },
    )
    output = tmp_path / "slo.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scale_reliability.slo_burn_rate",
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
    assert report["status"] == "within_budget"
    assert "burn_rate" in result.stdout
