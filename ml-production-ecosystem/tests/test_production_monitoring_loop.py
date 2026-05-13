from pathlib import Path
import json
import subprocess
import sys

from production_patterns.monitoring_loop import run_monitoring_loop


def test_run_monitoring_loop_records_passed_and_failed_checks(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"error_rate": 0.02, "p95_latency_ms": 120}))
    report_path = tmp_path / "reports" / "monitoring-loop.json"

    summary = run_monitoring_loop(
        metrics_path=metrics_path,
        output_path=report_path,
        maximums={"error_rate": 0.01, "p95_latency_ms": 250},
    )

    assert summary == {
        "status": "failed",
        "metrics_path": str(metrics_path),
        "checks": [
            {
                "check_name": "error_rate_maximum",
                "passed": False,
                "message": "error_rate 0.02 above maximum 0.01",
            },
            {
                "check_name": "p95_latency_ms_maximum",
                "passed": True,
                "message": "p95_latency_ms 120.0 within maximum 250.0",
            },
        ],
    }
    assert json.loads(report_path.read_text()) == summary


def test_production_monitoring_loop_cli_prints_summary(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    output_path = tmp_path / "monitoring-loop.json"
    metrics_path.write_text(json.dumps({"error_rate": 0.005}))

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "production_patterns.monitoring_loop",
            "--metrics-path",
            str(metrics_path),
            "--output-path",
            str(output_path),
            "--maximum",
            "error_rate=0.01",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["status"] == "passed"
    assert summary["checks"] == [
        {
            "check_name": "error_rate_maximum",
            "passed": True,
            "message": "error_rate 0.005 within maximum 0.01",
        }
    ]
    assert output_path.exists()
