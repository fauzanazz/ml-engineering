"""Multi-window SLO burn-rate alert simulation."""

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT_PATH = Path("03-scale-and-reliability/reports/burn-rate-alert.json")
DEFAULT_CRITICAL_SHORT_BURN_RATE = 14.0
DEFAULT_CRITICAL_LONG_BURN_RATE = 2.0
DEFAULT_WARNING_LONG_BURN_RATE = 1.0


def build_burn_rate_alert(
    short_window_report_path: Path,
    long_window_report_path: Path,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    critical_short_burn_rate: float = DEFAULT_CRITICAL_SHORT_BURN_RATE,
    critical_long_burn_rate: float = DEFAULT_CRITICAL_LONG_BURN_RATE,
    warning_long_burn_rate: float = DEFAULT_WARNING_LONG_BURN_RATE,
) -> dict[str, Any]:
    short_report = _read_json(short_window_report_path)
    long_report = _read_json(long_window_report_path)
    short_rates = _burn_rates(short_report)
    long_rates = _burn_rates(long_report)
    all_slos = sorted(set(short_rates) | set(long_rates))
    alerts = []

    for slo_name in all_slos:
        short_rate = short_rates.get(slo_name, 0.0)
        long_rate = long_rates.get(slo_name, 0.0)
        if short_rate >= critical_short_burn_rate and long_rate >= critical_long_burn_rate:
            alerts.append(_alert(slo_name, "critical", short_rate, long_rate, "fast and sustained budget burn"))
        elif long_rate > warning_long_burn_rate:
            alerts.append(_alert(slo_name, "warning", short_rate, long_rate, "sustained budget burn"))

    severity_order = {"critical": 2, "warning": 1}
    status = "healthy"
    if alerts:
        status = max((alert["severity"] for alert in alerts), key=lambda severity: severity_order[severity])

    report: dict[str, Any] = {
        "status": status,
        "short_window_report_path": str(short_window_report_path),
        "long_window_report_path": str(long_window_report_path),
        "thresholds": {
            "critical_short_burn_rate": critical_short_burn_rate,
            "critical_long_burn_rate": critical_long_burn_rate,
            "warning_long_burn_rate": warning_long_burn_rate,
        },
        "alerts": alerts,
        "short_window_burn_rates": short_rates,
        "long_window_burn_rates": long_rates,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _alert(slo_name: str, severity: str, short_rate: float, long_rate: float, reason: str) -> dict[str, Any]:
    return {
        "slo": slo_name,
        "severity": severity,
        "short_burn_rate": _format_rate(short_rate),
        "long_burn_rate": _format_rate(long_rate),
        "reason": reason,
    }


def _format_rate(value: float) -> float | str:
    return "inf" if value == float("inf") else round(value, 6)


def _burn_rates(report: dict[str, Any]) -> dict[str, float]:
    checks = report.get("checks", {})
    if not isinstance(checks, dict):
        return {}
    rates: dict[str, float] = {}
    for name, check in checks.items():
        if not isinstance(check, dict):
            continue
        rates[str(name)] = _to_float(check.get("burn_rate", 0.0))
    return rates


def _to_float(value: object) -> float:
    if value == "inf":
        return float("inf")
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        return data
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate multi-window SLO burn-rate alerting.")
    parser.add_argument("--short-window-report", type=Path, required=True)
    parser.add_argument("--long-window-report", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--critical-short-burn-rate", type=float, default=DEFAULT_CRITICAL_SHORT_BURN_RATE)
    parser.add_argument("--critical-long-burn-rate", type=float, default=DEFAULT_CRITICAL_LONG_BURN_RATE)
    parser.add_argument("--warning-long-burn-rate", type=float, default=DEFAULT_WARNING_LONG_BURN_RATE)
    args = parser.parse_args()

    report = build_burn_rate_alert(
        short_window_report_path=args.short_window_report,
        long_window_report_path=args.long_window_report,
        output_path=args.output_path,
        critical_short_burn_rate=args.critical_short_burn_rate,
        critical_long_burn_rate=args.critical_long_burn_rate,
        warning_long_burn_rate=args.warning_long_burn_rate,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
