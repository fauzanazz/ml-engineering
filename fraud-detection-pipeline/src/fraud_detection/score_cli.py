import argparse
import json
from pathlib import Path

import pandas as pd

from fraud_detection.inference import load_bundle, score_transactions


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--trust-artifact", action="store_true", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        payload = json.loads(args.input_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("input JSON must be one object")
        bundle = load_bundle(args.run_dir, trusted=args.trust_artifact)
        scored = score_transactions(bundle, pd.DataFrame([payload])).iloc[0]
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    print(
        json.dumps(
            {
                "fraud_probability": float(scored["fraud_probability"]),
                "decision_threshold": float(scored["decision_threshold"]),
                "prediction": int(scored["prediction"]),
            }
        )
    )
