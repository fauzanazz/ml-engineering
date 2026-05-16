from pathlib import Path
import json
from typing import Any

from ml_production_ecosystem.production_patterns.drift_detection import build_drift_report


class FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def json(self) -> dict[str, Any]:
        return self.payload


class FakeClient:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response

    def get(self, url: str, timeout: float) -> FakeResponse:
        assert url == "http://api.test/drift"
        return self.response


def test_drift_report_skips_without_base_url(tmp_path: Path) -> None:
    output_path = tmp_path / "drift-report.json"

    report = build_drift_report(None, output_path)

    assert report == {"status": "skipped", "reason": "base URL not provided"}
    assert json.loads(output_path.read_text()) == report


def test_drift_report_passes_under_threshold(tmp_path: Path) -> None:
    output_path = tmp_path / "drift-report.json"

    report = build_drift_report(
        "http://api.test",
        output_path,
        threshold=0.2,
        http_client=FakeClient(FakeResponse({"drift_score": 0.1, "sample_size": 10})),
    )

    assert report["status"] == "passed"
    assert report["score"] == 0.1
    assert report["threshold"] == 0.2
    assert json.loads(output_path.read_text()) == report


def test_drift_report_fails_over_threshold(tmp_path: Path) -> None:
    report = build_drift_report(
        "http://api.test",
        tmp_path / "drift-report.json",
        threshold=0.2,
        http_client=FakeClient(FakeResponse({"drift_score": 0.3})),
    )

    assert report["status"] == "failed"
    assert report["score"] == 0.3
