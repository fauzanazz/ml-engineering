from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx_whisper

from indonesian_banking_asr.synthetic.audit import write_jsonl

DEFAULT_MODEL = "mlx-community/whisper-large-v3-mlx"


def transcribe_manifest_rows(
    manifest_rows: list[dict],
    *,
    model: str = DEFAULT_MODEL,
    language: str = "id",
    limit: int | None = None,
) -> list[dict]:
    prediction_rows = []
    for row in manifest_rows[:limit]:
        result = mlx_whisper.transcribe(
            row["audio_path"],
            path_or_hf_repo=model,
            language=language,
            verbose=False,
        )
        prediction_rows.append(
            {
                "utterance_id": row["utterance_id"],
                "hypothesis": result["text"].strip(),
                "model": model,
                "language": language,
            }
        )
    return prediction_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe a project manifest with MLX Whisper.")
    parser.add_argument("--manifest-path", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--language", default="id")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    manifest_rows = _read_jsonl(args.manifest_path)
    prediction_rows = transcribe_manifest_rows(
        manifest_rows,
        model=args.model,
        language=args.language,
        limit=args.limit,
    )
    write_jsonl(args.output_path, prediction_rows)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


if __name__ == "__main__":
    main()
