from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import json
import subprocess
import sys
import threading

from ml_production_ecosystem.scale_reliability.alert_dispatch import dispatch_alerts


def _write_report(path: Path) -> Path:
    payload = {
        "alerts": [
            {
                "slo": "error_rate",
                "severity": "warning",
                "short_burn_rate": 1.4,
                "long_burn_rate": 1.2,
                "reason": "sustained budget burn",
            },
            {
                "slo": "availability",
                "severity": "critical",
                "short_burn_rate": 15.0,
                "long_burn_rate": 3.0,
                "reason": "fast and sustained budget burn",
            },
        ]
    }
    path.write_text(json.dumps(payload) + "\n")
    return path


def _start_webhook_server() -> tuple[str, list[dict[str, str]], threading.Thread, HTTPServer]:
    events: list[dict[str, str]] = []

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            payload = self.rfile.read(length).decode("utf-8") if length else ""
            events.append({"path": self.path, "body": payload})
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')

        def log_message(self, fmt: str, *args: object) -> None:  # pragma: no cover
            return

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    url = f"http://127.0.0.1:{server.server_port}"
    return url, events, thread, server


def test_dispatch_alerts_dry_run_marks_events_pending(tmp_path: Path) -> None:
    report_path = _write_report(tmp_path / "alert.json")
    output = tmp_path / "dispatch.json"

    report = dispatch_alerts(
        report_path,
        min_severity="warning",
        dry_run=True,
        webhook_url="https://example.invalid/hook",
        output_path=output,
    )

    assert report["status"] == "pending"
    assert report["candidate_count"] == 2
    assert report["dispatched_count"] == 0
    assert all(event["status"] == "pending" for event in report["events"])
    assert json.loads(output.read_text()) == report


def test_dispatch_alerts_min_severity_filters(tmp_path: Path, monkeypatch) -> None:
    report_path = _write_report(tmp_path / "alert.json")
    output = tmp_path / "dispatch.json"

    sent: list[dict] = []

    def fake_post(url: str, payload: dict, timeout_seconds: float = 2.0):
        sent.append({"url": url, "payload": payload})
        return True, "200"

    monkeypatch.setattr(
        "ml_production_ecosystem.scale_reliability.alert_dispatch._post_json",
        fake_post,
    )

    report = dispatch_alerts(
        report_path,
        min_severity="critical",
        webhook_url="https://example.invalid/hook",
        output_path=output,
    )

    assert report["status"] == "dispatched"
    assert report["candidate_count"] == 1
    assert report["dispatched_count"] == 1
    assert len(sent) == 1
    assert sent[0]["payload"]["alert"]["severity"] == "critical"


def test_dispatch_alerts_no_webhook_sets_pending(tmp_path: Path) -> None:
    report_path = _write_report(tmp_path / "alert.json")
    output = tmp_path / "dispatch.json"

    report = dispatch_alerts(
        report_path,
        webhook_url=None,
        dry_run=False,
        output_path=output,
    )

    assert report["status"] == "pending"
    assert report["webhook_url"] is None


def test_dispatch_alerts_cli_webhook_posts_events(tmp_path: Path) -> None:
    report_path = _write_report(tmp_path / "alert.json")
    output = tmp_path / "dispatch.json"

    webhook_url, events, thread, server = _start_webhook_server()
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "ml_production_ecosystem.scale_reliability.alert_dispatch",
                "--alert-report",
                str(report_path),
                "--min-severity",
                "warning",
                "--webhook-url",
                f"{webhook_url}/alerts",
                "--output-path",
                str(output),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    payload = json.loads(output.read_text())
    assert payload["status"] == "dispatched"
    assert payload["candidate_count"] == 2
    assert payload["dispatched_count"] == 2
    assert payload["events"][0]["dispatched"] is True
    assert payload["events"][0]["status"] == "dispatched"
    assert payload["events"][0]["webhook_url"] == f"{webhook_url}/alerts"
    assert payload["events"][1]["dispatched"] is True
    assert payload["events"][1]["status"] == "dispatched"
    assert len(events) == 2
    for event in events:
        body = json.loads(event["body"])
        assert body["event"] == "ml_reliability_burn_rate_alert"
        assert body["source"] == "scale-burn-rate-alert"

    assert "candidate_count" in result.stdout
