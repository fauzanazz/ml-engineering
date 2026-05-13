from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol


class TtsEngine(Protocol):
    engine_name: str
    sample_rate: int
    duration_sec: float

    def synthesize(self, text: str, output_path: Path) -> None:
        """Write speech audio for text to output_path."""


@dataclass(frozen=True)
class SyntheticToneTts:
    sample_rate: int = 8000
    duration_sec: float = 1.0
    engine_name: str = "synthetic-tone"

    def synthesize(self, text: str, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame_count = int(self.sample_rate * self.duration_sec)
        frequency_hz = 440 + (len(text) % 220)
        amplitude = 8000

        with wave.open(str(output_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            for frame_index in range(frame_count):
                sample = int(
                    amplitude
                    * math.sin(2 * math.pi * frequency_hz * frame_index / self.sample_rate)
                )
                wav_file.writeframesraw(sample.to_bytes(2, byteorder="little", signed=True))


def build_audio_manifest_rows(
    rows: Iterable[dict],
    audio_dir: Path,
    tts: TtsEngine,
) -> list[dict]:
    audio_rows = []
    for row in rows:
        utterance_id = row["utterance_id"]
        text = row.get("text")
        if not text:
            raise ValueError(f"row {utterance_id} missing text")

        audio_path = audio_dir / f"{utterance_id}.wav"
        tts.synthesize(text, audio_path)
        audio_rows.append(
            {
                **row,
                "audio_path": str(audio_path),
                "duration_sec": tts.duration_sec,
                "sample_rate": tts.sample_rate,
                "tts_engine": tts.engine_name,
            }
        )

    return audio_rows
