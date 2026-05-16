from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

from indonesian_banking_asr.synthetic.audit import write_jsonl

DEFAULT_SOURCE = "BabelSpeech/40hours_Indonesian_Colloquial_ASR_Speech_Dataset"


def build_babelspeech_manifest_rows(
    metadata_rows: list[dict],
    *,
    dataset_root: Path,
    extracted_audio_root: Path | None = None,
    source: str = DEFAULT_SOURCE,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    limit: int | None = None,
) -> list[dict]:
    selected_rows = metadata_rows[:limit]
    total_rows = len(selected_rows)
    train_count = int(total_rows * train_ratio)
    validation_count = int(total_rows * validation_ratio)

    manifest_rows = []
    for row_index, row in enumerate(selected_rows):
        relative_path = row["relative_path"]
        audio_path = _resolve_audio_path(dataset_root, relative_path, extracted_audio_root)
        manifest_rows.append(
            {
                "utterance_id": _utterance_id(row, row_index),
                "audio_path": str(audio_path),
                "text": row["text"].strip(),
                "split": _split_for_index(row_index, train_count, validation_count),
                "source": source,
                "duration": row.get("duration"),
                "confidence": row.get("confidence"),
                "snr": row.get("snr"),
                "dnsmos": row.get("dnsmos"),
            }
        )
    return manifest_rows


def extract_babelspeech_audio(zip_path: Path, output_dir: Path, manifest_rows: list[dict]) -> None:
    wanted_paths = {Path(row["audio_path"]).name for row in manifest_rows}
    output_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path) as wav_zip:
        for zip_info in wav_zip.infolist():
            zip_path_name = Path(zip_info.filename)
            if zip_path_name.name in wanted_paths and zip_info.filename.startswith("wav/"):
                wav_zip.extract(zip_info, output_dir.parent)


def read_metadata(path: Path) -> list[dict]:
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise ValueError("BabelSpeech metadata must be JSON list")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert BabelSpeech metadata to project JSONL manifest.")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--metadata-path", type=Path)
    parser.add_argument("--extracted-audio-root", type=Path)
    parser.add_argument("--extract-audio", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    metadata_path = args.metadata_path or args.dataset_root / "audio_info.json"
    extracted_audio_root = args.extracted_audio_root or args.output_path.parent / "wav"
    rows = build_babelspeech_manifest_rows(
        read_metadata(metadata_path),
        dataset_root=args.dataset_root,
        extracted_audio_root=extracted_audio_root if args.extract_audio else args.extracted_audio_root,
        limit=args.limit,
    )
    if args.extract_audio:
        extract_babelspeech_audio(args.dataset_root / "wav.zip", extracted_audio_root, rows)
    write_jsonl(args.output_path, rows)


def _resolve_audio_path(dataset_root: Path, relative_path: str, extracted_audio_root: Path | None) -> Path:
    if extracted_audio_root is None:
        return dataset_root / relative_path
    return extracted_audio_root / Path(relative_path).name


def _utterance_id(row: dict, row_index: int) -> str:
    filename = str(row.get("filename") or Path(row["relative_path"]).name)
    return Path(filename).stem or f"babelspeech-{row_index:06d}"


def _split_for_index(row_index: int, train_count: int, validation_count: int) -> str:
    if row_index < train_count:
        return "train"
    if row_index < train_count + validation_count:
        return "validation"
    return "test"
