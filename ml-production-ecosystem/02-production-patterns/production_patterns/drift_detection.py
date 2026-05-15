"""Standalone drift detection report for local serving endpoints."""

from pathlib import Path
import argparse
import json
from typing import Any, Protocol

import httpx

DEFAULT_OUTPUT_PATH = Path("02-production-patterns/reports/drift-report.json")


class HttpResponse(Protocol):
    status_code: int

    def json(self) -> dict[str, Any]: ...


class HttpClient(Protocol):
    def get(self, url: str, timeout: float) -> HttpResponse: ...


def build_drift_report(
    base_url: str | None,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    threshold: float = 0.2,
    http_client: HttpClient | None = None,
    timeout: float = 5.0,
) -> dict[str, object]:
    if base_url is None:
        report: dict[str, object] = {"status": "skipped", "reason": "base URL not provided"}
    else:
        client = http_client or httpx.Client()
        try:
            response = client.get(f"{base_url.rstrip('/')}/drift", timeout=timeout)
            if response.status_code >= 400:
                raise RuntimeError(f"HTTP {response.status_code}")
            payload = response.json()
            score = float(payload.get("drift_score", 0.0))
            report = {
                "status": "passed" if score <= threshold else "failed",
                "base_url": base_url,
                "score": score,
                "threshold": threshold,
                "payload": payload,
            }
        except Exception as error:
            report = {"status": "failed", "base_url": base_url, "threshold": threshold, "error": str(error)}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Write standalone drift detection report.")
    parser.add_argument("--base-url")
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    report = build_drift_report(args.base_url, args.output_path, args.threshold)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
