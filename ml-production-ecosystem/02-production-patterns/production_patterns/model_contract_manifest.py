"""Model I/O contract manifest validation for generic lifecycle flows."""

from pathlib import Path
import argparse
import json
from typing import Any

import yaml

DEFAULT_OUTPUT_PATH = Path("02-production-patterns/reports/model-contract-manifest.json")
REQUIRED_FIELDS = ("input_schema_uri", "output_schema_uri", "task_type", "prediction_key")


def _load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open() as file:
        config = yaml.safe_load(file)
    if isinstance(config, dict):
        return config
    return {}


def _contract_config(config: dict[str, Any]) -> dict[str, Any]:
    contract = config.get("model_contract", {})
    if isinstance(contract, dict):
        return contract
    return {}


def _schema_status(path: str) -> dict[str, object]:
    if not path:
        return {"path": path, "exists": False, "valid_json": False}
    schema_path = Path(path)
    if not schema_path.exists():
        return {"path": path, "exists": False, "valid_json": False}
    try:
        payload = json.loads(schema_path.read_text())
    except json.JSONDecodeError:
        return {"path": path, "exists": True, "valid_json": False}
    return {
        "path": path,
        "exists": True,
        "valid_json": True,
        "title": payload.get("title") if isinstance(payload, dict) else None,
    }


def build_model_contract_manifest(config_path: Path, output_path: Path = DEFAULT_OUTPUT_PATH) -> dict[str, object]:
    contract = _contract_config(_load_config(config_path))
    missing = [field for field in REQUIRED_FIELDS if not contract.get(field)]
    input_status = _schema_status(str(contract.get("input_schema_uri", "")))
    output_status = _schema_status(str(contract.get("output_schema_uri", "")))
    schema_ok = bool(input_status["valid_json"] and output_status["valid_json"])

    manifest = {
        "status": "ready" if not missing and schema_ok else "blocked",
        "missing_fields": missing,
        "contract": {field: contract.get(field) for field in REQUIRED_FIELDS},
        "schemas": {
            "input": input_status,
            "output": output_status,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate model I/O contract manifest from config.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    manifest = build_model_contract_manifest(args.config, args.output_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
