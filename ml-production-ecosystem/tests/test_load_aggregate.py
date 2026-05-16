"""Distributed load shard aggregation tests."""

from pathlib import Path
import json
import subprocess
import sys

import pytest

from ml_production_ecosystem.scale_reliability.load_aggregate import aggregate_load_reports


def _load_report(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def test_aggregate_load_reports_passed_shards(tmp_path: Path) -> None:
    first = _load_report(
        tmp_path / "shard-1.json",
        {
            "status": "passed",
            "request_count": 10,
            "success_count": 10,
            "error_count": 0,
            "attempt_count": 10,
            "retry_attempt_count": 0,
            "retried_request_count": 0,
            "retry_success_count": 0,
            "retry_exhausted_count": 0,
            "latency_ms": {"min": 5.0, "p50": 10.0, "p95": 20.0, "max": 25.0},
            "errors": {},
        },
    )
    second = _load_report(
        tmp_path / "shard-2.json",
        {
            "status": "passed",
            "request_count": 20,
            "success_count": 20,
            "error_count": 0,
            "attempt_count": 20,
            "retry_attempt_count": 0,
            "retried_request_count": 0,
            "retry_success_count": 0,
            "retry_exhausted_count": 0,
            "latency_ms": {"min": 7.0, "p50": 15.0, "p95": 30.0, "max": 40.0},
            "errors": {},
        },
    )

    summary = aggregate_load_reports([first, second], tmp_path / "aggregate.json")

    assert summary["status"] == "passed"
    assert summary["shard_count"] == 2
    assert summary["request_count"] == 30
    assert summary["success_count"] == 30
    assert summary["error_rate"] == 0.0
    assert summary["latency_ms"] == {"min": 5.0, "p50": 15.0, "p95": 30.0, "max": 40.0}
    assert json.loads((tmp_path / "aggregate.json").read_text()) == summary


def test_aggregate_load_reports_degraded_with_errors(tmp_path: Path) -> None:
    first = _load_report(
        tmp_path / "shard-1.json",
        {
            "status": "degraded",
            "request_count": 10,
            "success_count": 8,
            "error_count": 2,
            "attempt_count": 12,
            "retry_attempt_count": 2,
            "retried_request_count": 2,
            "retry_success_count": 1,
            "retry_exhausted_count": 1,
            "latency_ms": {"min": 5.0, "p50": 10.0, "p95": 20.0, "max": 25.0},
            "errors": {"timeout": 1, "http_500": 1},
        },
    )
    second = _load_report(
        tmp_path / "shard-2.json",
        {
            "status": "passed",
            "request_count": 10,
            "success_count": 10,
            "error_count": 0,
            "attempt_count": 10,
            "retry_attempt_count": 0,
            "retried_request_count": 0,
            "retry_success_count": 0,
            "retry_exhausted_count": 0,
            "latency_ms": {"min": 4.0, "p50": 8.0, "p95": 12.0, "max": 18.0},
            "errors": {},
        },
    )

    summary = aggregate_load_reports([first, second], tmp_path / "aggregate.json")

    assert summary["status"] == "degraded"
    assert summary["error_count"] == 2
    assert summary["error_rate"] == 0.1
    assert summary["errors"] == {"timeout": 1, "http_500": 1}
    assert summary["retry_attempt_count"] == 2


def test_aggregate_load_reports_rejects_empty_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one report path is required"):
        aggregate_load_reports([], tmp_path / "aggregate.json")


def test_aggregate_load_reports_cli_writes_summary(tmp_path: Path) -> None:
    report = _load_report(
        tmp_path / "shard.json",
        {
            "status": "passed",
            "request_count": 1,
            "success_count": 1,
            "error_count": 0,
            "attempt_count": 1,
            "retry_attempt_count": 0,
            "retried_request_count": 0,
            "retry_success_count": 0,
            "retry_exhausted_count": 0,
            "latency_ms": {"min": 1.0, "p50": 1.0, "p95": 1.0, "max": 1.0},
            "errors": {},
        },
    )
    output = tmp_path / "aggregate.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ml_production_ecosystem.scale_reliability.load_aggregate",
            "--report",
            str(report),
            "--output-path",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    summary = json.loads(output.read_text())
    assert summary["status"] == "passed"
    assert "shard_count" in result.stdout
