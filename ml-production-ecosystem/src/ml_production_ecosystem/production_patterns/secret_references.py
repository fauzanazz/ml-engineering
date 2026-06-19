"""Validate committed configs store secret references, not secret values."""

from pathlib import Path
import argparse
import json
from typing import Any

import yaml

DEFAULT_ROOT = Path(".")
DEFAULT_OUTPUT_PATH = Path("artifacts/reports/production-patterns/secret-reference-validation.json")
SCAN_DIRS = (Path("configs"), Path("configs/platform"))
SCANNED_SUFFIXES = {".yaml", ".yml", ".json"}
FORBIDDEN_SECRET_VALUE_KEYS = {
    "access_key",
    "api_key",
    "client_secret",
    "password",
    "private_key",
    "secret",
    "secret_value",
    "token_value",
    "value",
}

def _read_structured_documents(path: Path) -> list[Any]:
    if path.suffix == ".json":
        return [json.loads(path.read_text())]
    return list(yaml.safe_load_all(path.read_text()))

def _scan_value(value: Any, location: str) -> list[str]:
    violations = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            child_location = f"{location}.{key_text}" if location else key_text
            if key_text.lower() in FORBIDDEN_SECRET_VALUE_KEYS:
                violations.append(child_location)
            violations.extend(_scan_value(child, child_location))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            violations.extend(_scan_value(child, f"{location}[{index}]"))
    return violations

def _candidate_files(root: Path) -> list[Path]:
    files = []
    for scan_dir in SCAN_DIRS:
        directory = root / scan_dir
        if not directory.exists():
            continue
        files.extend(path for path in directory.rglob("*") if path.suffix in SCANNED_SUFFIXES)
    return sorted(files)

def validate_secret_references(
    root: Path = DEFAULT_ROOT,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, object]:
    violations = []
    for path in _candidate_files(root):
        relative_path = path.relative_to(root)
        try:
            payloads = _read_structured_documents(path)
        except Exception as error:
            violations.append({"path": str(relative_path), "location": "<parse>", "reason": str(error)})
            continue
        for payload in payloads:
            if payload is None:
                continue
            for location in _scan_value(payload, ""):
                violations.append(
                    {
                        "path": str(relative_path),
                        "location": location,
                        "reason": "forbidden secret value key",
                    }
                )

    report = {
        "status": "passed" if not violations else "failed",
        "scanned_dirs": [str(path) for path in SCAN_DIRS],
        "scanned_suffixes": sorted(SCANNED_SUFFIXES),
        "violations": violations,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate secret references without committed secret values.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    report = validate_secret_references(args.root, args.output_path)
    print(json.dumps(report, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
