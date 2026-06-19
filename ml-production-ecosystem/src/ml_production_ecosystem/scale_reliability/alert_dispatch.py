"""Dispatch burn-rate alerts to downstream runbooks or incident endpoints."""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT_PATH = Path("artifacts/reports/scale-reliability/alert-dispatch.json")
SEVERITY_LEVELS = {
    "healthy": 0,
    "warning": 1,
    "critical": 2,
}


def _read_report(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        return data
    return {}


def _alerts(report: dict[str, Any]) -> list[dict[str, Any]]:
    raw_alerts = report.get("alerts", [])
    if not isinstance(raw_alerts, list):
        return []

    normalized: list[dict[str, Any]] = []
    for alert in raw_alerts:
        if not isinstance(alert, dict):
            continue
        normalized.append(
            {
                "slo": str(alert.get("slo", "unknown")),
                "severity": str(alert.get("severity", "warning")).lower(),
                "short_burn_rate": alert.get("short_burn_rate"),
                "long_burn_rate": alert.get("long_burn_rate"),
                "reason": str(alert.get("reason", "")),
            }
        )
    return normalized


def _filter_alerts(alerts: list[dict[str, Any]], min_severity: str) -> list[dict[str, Any]]:
    min_level = SEVERITY_LEVELS.get(min_severity, 1)
    return [
        alert
        for alert in alerts
        if SEVERITY_LEVELS.get(str(alert.get("severity")).lower(), 0) >= min_level
    ]


def _post_json(url: str, payload: dict[str, Any], timeout_seconds: float = 2.0) -> tuple[bool, str | None]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response.read()
            return True, str(response.status)
    except Exception as exc:  # pragma: no cover - exercised through tests and exception surface
        if isinstance(exc, urllib.error.HTTPError):
            return False, f"{exc.code}: {exc.reason}"
        return False, str(exc)


def dispatch_alerts(
    alert_report_path: Path,
    *,
    webhook_url: str | None = None,
    min_severity: str = "warning",
    dry_run: bool = False,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    timeout_seconds: float = 2.0,
) -> dict[str, Any]:
    report = _read_report(alert_report_path)
    candidate_alerts = _filter_alerts(_alerts(report), min_severity=min_severity)

    if not candidate_alerts:
        summary = {
            "status": "idle",
            "reason": "no alerts above severity threshold",
            "candidate_count": 0,
            "dispatched_count": 0,
            "failed_count": 0,
            "events": [],
            "dry_run": dry_run,
            "min_severity": min_severity,
            "webhook_url": webhook_url,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        return summary

    dispatch_targets = webhook_url or None
    events: list[dict[str, Any]] = []
    dispatched = 0
    failed = 0

    send_events = bool(dispatch_targets) and not dry_run

    for alert in candidate_alerts:
        event = {
            "alert": alert,
            "webhook_url": dispatch_targets,
            "dispatched": False,
            "status": "pending",
            "error": None,
        }

        if send_events:
            ok, status_or_error = _post_json(
                dispatch_targets,
                {
                    "event": "ml_reliability_burn_rate_alert",
                    "alert": alert,
                    "source": "scale-burn-rate-alert",
                },
                timeout_seconds=timeout_seconds,
            )
            if ok:
                event["dispatched"] = True
                event["status"] = "dispatched"
                event["status_code"] = status_or_error
                dispatched += 1
            else:
                event["status"] = "failed"
                event["error"] = status_or_error
                failed += 1
        else:
            event["status"] = "pending"

        events.append(event)

    if failed:
        status = "partial" if dispatched else "failed"
    elif send_events:
        status = "dispatched"
    else:
        status = "pending"
    summary = {
        "status": status,
        "candidate_count": len(candidate_alerts),
        "dispatched_count": dispatched,
        "failed_count": failed,
        "dry_run": dry_run,
        "min_severity": min_severity,
        "webhook_url": dispatch_targets,
        "events": events,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Dispatch burn-rate alerts to webhook endpoints.")
    parser.add_argument("--alert-report", type=Path, required=True)
    parser.add_argument("--min-severity", default="warning", choices=tuple(SEVERITY_LEVELS.keys()))
    parser.add_argument("--webhook-url", default=os.getenv("ALERT_WEBHOOK_URL"))
    parser.add_argument("--dry-run", action="store_true", help="Do not send HTTP calls, only mark alerts pending")
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    report = dispatch_alerts(
        alert_report_path=args.alert_report,
        webhook_url=args.webhook_url,
        min_severity=args.min_severity,
        dry_run=args.dry_run,
        output_path=args.output_path,
        timeout_seconds=args.timeout,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
