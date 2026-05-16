"""Local traffic-splitting canary router simulation."""

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT_PATH = Path("artifacts/reports/production-patterns/local-canary-router.json")

def build_canary_routes(
    decision_path: Path,
    stable_model_id: str,
    candidate_model_id: str,
    request_count: int = 100,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    decision = _read_json(decision_path)
    canary_percent = _valid_percent(decision.get("canary_percent", 0))
    candidate_percent = canary_percent if _should_route_candidate(decision) else 0
    stable_percent = 100 - candidate_percent
    routes = _build_routes(stable_model_id, candidate_model_id, stable_percent, candidate_percent, request_count)
    report: dict[str, Any] = {
        "status": "passed",
        "mode": "local-traffic-splitting-simulation",
        "decision_path": str(decision_path),
        "decision": decision.get("decision", "unknown"),
        "request_count": request_count,
        "stable_model_id": stable_model_id,
        "candidate_model_id": candidate_model_id,
        "stable_percent": stable_percent,
        "candidate_percent": candidate_percent,
        "routes": routes,
        "rollback_required": candidate_percent == 0 and decision.get("decision") == "rollback",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report

def _should_route_candidate(decision: dict[str, Any]) -> bool:
    return decision.get("decision") == "promote" and decision.get("status") == "passed"

def _build_routes(
    stable_model_id: str,
    candidate_model_id: str,
    stable_percent: int,
    candidate_percent: int,
    request_count: int,
) -> list[dict[str, Any]]:
    if request_count <= 0:
        raise ValueError("request_count must be greater than 0")
    candidate_requests = round(request_count * candidate_percent / 100)
    stable_requests = request_count - candidate_requests
    return [
        {"model_id": stable_model_id, "role": "stable", "traffic_percent": stable_percent, "requests": stable_requests},
        {"model_id": candidate_model_id, "role": "candidate", "traffic_percent": candidate_percent, "requests": candidate_requests},
    ]

def _valid_percent(value: object) -> int:
    percent = int(value)
    if percent < 0 or percent > 100:
        raise ValueError("canary_percent must be between 0 and 100")
    return percent

def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data

def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate local canary traffic splitting from release decision.")
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--stable-model-id", required=True)
    parser.add_argument("--candidate-model-id", required=True)
    parser.add_argument("--request-count", type=int, default=100)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    report = build_canary_routes(
        decision_path=args.decision,
        stable_model_id=args.stable_model_id,
        candidate_model_id=args.candidate_model_id,
        request_count=args.request_count,
        output_path=args.output_path,
    )
    print(json.dumps(report, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
