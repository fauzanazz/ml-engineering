from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify best validation log RMSE against a required threshold.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--threshold", type=float, default=0.00044)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics_paths = sorted(args.artifacts_dir.glob("*/metrics.json"))
    if not metrics_paths:
        raise SystemExit(f"No metrics.json files found under {args.artifacts_dir}")

    scored_runs = []
    for metrics_path in metrics_paths:
        metrics = json.loads(metrics_path.read_text())
        if "log_rmse" in metrics:
            scored_runs.append((float(metrics["log_rmse"]), metrics_path.parent))

    if not scored_runs:
        raise SystemExit("No metrics.json file contains log_rmse")

    best_score, best_run = min(scored_runs, key=lambda item: item[0])
    passed = best_score <= args.threshold
    print(f"best_run={best_run}")
    print(f"best_log_rmse={best_score:.15f}")
    print(f"threshold={args.threshold:.15f}")
    print(f"passed={str(passed).lower()}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
