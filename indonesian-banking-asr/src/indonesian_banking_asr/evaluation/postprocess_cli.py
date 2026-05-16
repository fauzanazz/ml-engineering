from __future__ import annotations

import argparse
import json
from pathlib import Path

from indonesian_banking_asr.evaluation.postprocess import postprocess_transcript
from indonesian_banking_asr.synthetic.audit import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-process ASR prediction hypotheses.")
    parser.add_argument("--predictions-path", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    args = parser.parse_args()

    rows = _read_jsonl(args.predictions_path)
    write_jsonl(args.output_path, [_postprocess_prediction_row(row) for row in rows])


def _postprocess_prediction_row(row: dict) -> dict:
    raw_hypothesis = row["hypothesis"]
    return {
        **row,
        "raw_hypothesis": raw_hypothesis,
        "hypothesis": postprocess_transcript(raw_hypothesis),
        "postprocess": "banking_entity_v1",
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


if __name__ == "__main__":
    main()
