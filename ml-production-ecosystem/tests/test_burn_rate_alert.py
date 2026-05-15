"""Multi-window burn-rate alert tests."""

from pathlib import Path
import json
import subprocess
import sys

from scale_reliability.burn_rate_alert import build_burn_rate_alert


def _slo_report(path: Path, rates: dict[str, float | str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"checks": {name: {"burn_rate": rate} for name, rate in rates.items()}}))
    return path


def test_burn_rate_alert_healthy_when_windows_within_budget(tmp_path: Path) -> None:
    short = _slo_report(tmp_path / "short.json", {"error_rate": 0.5})
    long = _slo_report(tmp_path / "long.json", {"error_rate": 0.8})

    report = build_burn_rate_alert(short, long, tmp_path / "alert.json")

    assert report["status"] == "healthy"
    assert report["alerts"] == []
    assert json.loads((tmp_path / "alert.json").read_text()) == report


def test_burn_rate_alert_warns_on_sustained_long_window_burn(tmp_path: Path) -> None:
    short = _slo_report(tmp_path / "short.json", {"latency_p95_ms": 2.0})
    long = _slo_report(tmp_path / "long.json", {"latency_p95_ms": 1.2})

    report = build_burn_rate_alert(short, long, tmp_path / "alert.json")

    assert report["status"] == "warning"
    assert report["alerts"] == [
        {
            "slo": "latency_p95_ms",
            "severity": "warning",
            "short_burn_rate": 2.0,
            "long_burn_rate": 1.2,
            "reason": "sustained budget burn",
        }
    ]


def test_burn_rate_alert_critical_on_fast_and_sustained_burn(tmp_path: Path) -> None:
    short = _slo_report(tmp_path / "short.json", {"availability": 15.0})
    long = _slo_report(tmp_path / "long.json", {"availability": 2.5})

    report = build_burn_rate_alert(short, long, tmp_path / "alert.json")

    assert report["status"] == "critical"
    assert report["alerts"][0]["severity"] == "critical"
    assert report["alerts"][0]["reason"] == "fast and sustained budget burn"


def test_burn_rate_alert_handles_infinite_burn_rate(tmp_path: Path) -> None:
    short = _slo_report(tmp_path / "short.json", {"error_rate": "inf"})
    long = _slo_report(tmp_path / "long.json", {"error_rate": 3.0})

    report = build_burn_rate_alert(short, long, tmp_path / "alert.json")

    assert report["status"] == "critical"
    assert report["alerts"][0]["short_burn_rate"] == "inf"


def test_burn_rate_alert_cli_writes_report(tmp_path: Path) -> None:
    short = _slo_report(tmp_path / "short.json", {"error_rate": 0.0})
    long = _slo_report(tmp_path / "long.json", {"error_rate": 0.0})
    output = tmp_path / "alert.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scale_reliability.burn_rate_alert",
            "--short-window-report",
            str(short),
            "--long-window-report",
            str(long),
            "--output-path",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    report = json.loads(output.read_text())
    assert report["status"] == "healthy"
    assert "critical_short_burn_rate" in result.stdout
