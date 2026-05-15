from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .pipeline import run_smoke_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local DeepSeek-R1-style reasoning post-training smoke pipeline.")
    parser.add_argument("--config", required=True, help="YAML config path")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    result = run_smoke_pipeline(config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
