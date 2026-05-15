"""Local canary traffic router tests."""

from pathlib import Path
import json
import subprocess
import sys

from production_patterns.canary_router import build_canary_routes

def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload))
    return path

def test_canary_router_splits_promoted_candidate_traffic(tmp_path: Path) -> None:
    decision = _write_json(tmp_path / "decision.json", {"status": "passed", "decision": "promote", "canary_percent": 25})

    report = build_canary_routes(decision, "stable-v1", "candidate-v2", 40, tmp_path / "router.json")

    assert report["status"] == "passed"
    assert report["mode"] == "local-traffic-splitting-simulation"
    assert report["stable_percent"] == 75
    assert report["candidate_percent"] == 25
    assert report["routes"] == [
        {"model_id": "stable-v1", "role": "stable", "traffic_percent": 75, "requests": 30},
        {"model_id": "candidate-v2", "role": "candidate", "traffic_percent": 25, "requests": 10},
    ]
    assert json.loads((tmp_path / "router.json").read_text()) == report

def test_canary_router_sends_all_traffic_to_stable_on_rollback(tmp_path: Path) -> None:
    decision = _write_json(tmp_path / "decision.json", {"status": "blocked", "decision": "rollback", "canary_percent": 10})

    report = build_canary_routes(decision, "stable-v1", "candidate-v2", 20, tmp_path / "router.json")

    assert report["rollback_required"] is True
    assert report["stable_percent"] == 100
    assert report["candidate_percent"] == 0
    assert report["routes"][0]["requests"] == 20
    assert report["routes"][1]["requests"] == 0

def test_canary_router_cli_writes_report(tmp_path: Path) -> None:
    decision = _write_json(tmp_path / "decision.json", {"status": "passed", "decision": "promote", "canary_percent": 10})
    output_path = tmp_path / "router.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "production_patterns.canary_router",
            "--decision",
            str(decision),
            "--stable-model-id",
            "stable-v1",
            "--candidate-model-id",
            "candidate-v2",
            "--output-path",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    report = json.loads(output_path.read_text())
    assert report["candidate_percent"] == 10
    assert "local-traffic-splitting-simulation" in result.stdout
