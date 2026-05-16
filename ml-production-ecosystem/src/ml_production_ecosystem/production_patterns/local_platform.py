"""Apply local platform filesystem resources without cloud services."""

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

DEFAULT_PROJECT_ROOT = Path(".")
DEFAULT_OUTPUT_PATH = Path("artifacts/reports/production-patterns/local-platform-apply.json")
ADAPTER_PATH = Path("configs/platform/adapters/local/adapter.py")


def apply_local_platform(
    project_root: Path = DEFAULT_PROJECT_ROOT,
    environment: str = "development",
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    adapter_module = _load_local_adapter(project_root / ADAPTER_PATH)
    config = adapter_module.LocalAdapterConfig(project_root=project_root)
    adapter = adapter_module.LocalProviderAdapter(config)
    summary = adapter.ensure_resources(environment)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _load_local_adapter(adapter_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("local_platform_adapter", adapter_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load local adapter: {adapter_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description="Create local platform resources from local adapter plan.")
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--environment", default="development")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    summary = apply_local_platform(args.project_root, args.environment, args.output_path)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
