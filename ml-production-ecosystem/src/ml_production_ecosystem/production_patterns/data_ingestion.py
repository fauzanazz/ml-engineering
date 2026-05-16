"""Generic local dataset ingestion manifest for production lifecycle flows."""

from pathlib import Path
import argparse
import json
from typing import Any

import yaml

DEFAULT_OUTPUT_PATH = Path("02-production-patterns/reports/dataset-manifest.json")


def _load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open() as file:
        config = yaml.safe_load(file)
    if isinstance(config, dict):
        return config
    return {}


def _dataset_config(config: dict[str, Any]) -> dict[str, Any]:
    dataset = config.get("dataset", {})
    if isinstance(dataset, dict):
        return dataset
    return {}


def _source_paths(dataset: dict[str, Any]) -> dict[str, str]:
    return {key: str(value) for key, value in dataset.items() if key.endswith("_path")}


def _dataset_name(config: dict[str, Any], dataset: dict[str, Any]) -> str:
    explicit_name = dataset.get("name")
    if explicit_name:
        return str(explicit_name)
    pipeline = config.get("pipeline", {})
    if isinstance(pipeline, dict) and pipeline.get("name"):
        return str(pipeline["name"])
    return "local-dataset"


def build_dataset_manifest(config_path: Path, output_path: Path = DEFAULT_OUTPUT_PATH) -> dict[str, object]:
    config = _load_config(config_path)
    dataset = _dataset_config(config)
    sources = _source_paths(dataset)
    missing = [path for path in sources.values() if not Path(path).exists()]
    version = dataset.get("version")
    schema_uri = str(dataset.get("schema_uri", "schema://not-declared"))

    manifest = {
        "status": "ready" if not missing and bool(sources) else "blocked",
        "name": _dataset_name(config, dataset),
        "version": str(version) if version is not None else None,
        "uri": str(dataset.get("uri", config_path.parent)),
        "schema_uri": schema_uri,
        "sources": sources,
        "missing_sources": missing,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build provider-neutral local dataset manifest.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    manifest = build_dataset_manifest(args.config, args.output_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
