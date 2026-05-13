from __future__ import annotations

import argparse
import json
from pathlib import Path

from indonesian_banking_asr.evaluation.metrics import evaluate_predictions
from indonesian_banking_asr.synthetic.audit import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ASR predictions against a dataset manifest.")
    parser.add_argument("--manifest-path", required=True, type=Path)
    parser.add_argument("--predictions-path", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    args = parser.parse_args()

    manifest_rows = _read_jsonl(args.manifest_path)
    prediction_rows = _read_jsonl(args.predictions_path)
    summary = evaluate_predictions(manifest_rows, prediction_rows)
    write_jsonl(args.output_path, [summary])


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


if __name__ == "__main__":
    main()
