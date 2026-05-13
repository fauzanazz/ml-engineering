from __future__ import annotations

import wave
from pathlib import Path
from typing import Iterable


def validate_audio_manifest_rows(rows: Iterable[dict]) -> dict:
    errors = []
    checked_rows = 0
    invalid_utterance_ids = set()

    for row in rows:
        checked_rows += 1
        utterance_id = row["utterance_id"]
        audio_path = Path(row.get("audio_path", ""))
        if not audio_path.exists():
            errors.append(
                {
                    "utterance_id": utterance_id,
                    "field": "audio_path",
                    "message": "audio file missing",
                }
            )
            invalid_utterance_ids.add(utterance_id)
            continue

        try:
            with wave.open(str(audio_path), "rb") as wav_file:
                actual_sample_rate = wav_file.getframerate()
                frame_count = wav_file.getnframes()
                audio_bytes = wav_file.readframes(frame_count)
        except wave.Error:
            errors.append(
                {
                    "utterance_id": utterance_id,
                    "field": "audio_path",
                    "message": "invalid wav file",
                }
            )
            invalid_utterance_ids.add(utterance_id)
            continue

        if not any(audio_bytes):
            errors.append(
                {
                    "utterance_id": utterance_id,
                    "field": "audio_path",
                    "message": "audio is silent",
                }
            )
            invalid_utterance_ids.add(utterance_id)

        expected_sample_rate = row.get("sample_rate")
        if actual_sample_rate != expected_sample_rate:
            errors.append(
                {
                    "utterance_id": utterance_id,
                    "field": "sample_rate",
                    "message": f"expected {expected_sample_rate}, got {actual_sample_rate}",
                }
            )
            invalid_utterance_ids.add(utterance_id)

        expected_duration = row.get("duration_sec")
        actual_duration = round(frame_count / actual_sample_rate, 2)
        if actual_duration != expected_duration:
            errors.append(
                {
                    "utterance_id": utterance_id,
                    "field": "duration_sec",
                    "message": f"expected {expected_duration}, got {actual_duration}",
                }
            )
            invalid_utterance_ids.add(utterance_id)

    invalid_rows = len(invalid_utterance_ids)
    return {
        "checked_rows": checked_rows,
        "valid_rows": checked_rows - invalid_rows,
        "invalid_rows": invalid_rows,
        "errors": errors,
    }
