from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_local_deployment(
    model_uri: str = "models/local-dev",
    output_path: Path = Path("artifacts/reports/deployment.json"),
) -> dict[str, Any]:
    report = {
        "status": "completed",
        "project": "{{project_name}}",
        "package": "{{package_name}}",
        "backend": "{{backend}}",
        "provider": "{{provider}}",
        "model_uri": model_uri,
        "endpoint": "local://{{package_name}}/{{backend}}",
        "deployed_at": datetime.now(UTC).isoformat(),
    }
    _write_json(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local deployment evidence for {{project_name}}.")
    parser.add_argument("--model-uri", default="models/local-dev")
    parser.add_argument("--output-path", type=Path, default=Path("artifacts/reports/deployment.json"))
    args = parser.parse_args()
    print(json.dumps(run_local_deployment(args.model_uri, args.output_path), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
