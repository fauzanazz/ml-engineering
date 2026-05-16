"""Validate provider-specific code stays behind platform adapters."""

from pathlib import Path
import argparse
import json
import re

DEFAULT_ROOT = Path(".")
DEFAULT_OUTPUT_PATH = Path("artifacts/reports/production-patterns/provider-boundary-validation.json")
FORBIDDEN_PROVIDER_IMPORT = re.compile(
    r"^\s*(?:from|import)\s+"
    r"(boto3|botocore|google\.cloud|azure|kubernetes|terraform)"
    r"(?:\b|\.)",
    re.MULTILINE,
)
ALLOWED_PROVIDER_PATHS = (
    Path("configs/platform/adapters"),
    Path("configs/platform"),
)
IGNORED_PATH_PARTS = {".git", ".venv", "__pycache__"}

def _is_ignored(path: Path) -> bool:
    return any(part in IGNORED_PATH_PARTS for part in path.parts)

def _is_allowed_provider_path(relative_path: Path) -> bool:
    return any(
        relative_path == allowed_path or allowed_path in relative_path.parents
        for allowed_path in ALLOWED_PROVIDER_PATHS
    )

def find_provider_boundary_violations(root: Path = DEFAULT_ROOT) -> list[str]:
    violations = []
    for source_path in sorted(root.rglob("*.py")):
        relative_path = source_path.relative_to(root)
        if _is_ignored(relative_path) or _is_allowed_provider_path(relative_path):
            continue
        if FORBIDDEN_PROVIDER_IMPORT.search(source_path.read_text()):
            violations.append(str(relative_path))
    return violations

def validate_provider_boundaries(
    root: Path = DEFAULT_ROOT,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, object]:
    violations = find_provider_boundary_violations(root)
    report = {
        "status": "passed" if not violations else "failed",
        "root": str(root),
        "allowed_provider_paths": [str(path) for path in ALLOWED_PROVIDER_PATHS],
        "violations": violations,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate provider-specific imports stay behind adapters.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    report = validate_provider_boundaries(args.root, args.output_path)
    print(json.dumps(report, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
