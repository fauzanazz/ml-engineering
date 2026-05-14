from __future__ import annotations

import asyncio
import base64
import math
import subprocess
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

import edge_tts

from indonesian_banking_asr.synthetic.gemini import GeminiTransport, RetryableGeminiError, UrllibGeminiTransport
from indonesian_banking_asr.synthetic.rate_limit import RateLimiter


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


@dataclass(frozen=True)
class EdgeTts:
    voice_name: str = "id-ID-ArdiNeural"
    sample_rate: int = 16000
    duration_sec: float = 0.0
    engine_name: str = "edge-tts"
    communicate_factory: Callable[[str, str], Any] = edge_tts.Communicate
    convert_to_wav: Callable[[Path, Path, int], None] | None = None

    def synthesize(self, text: str, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mp3_path = output_path.with_suffix(".mp3")
        communicate = self.communicate_factory(text, self.voice_name)
        asyncio.run(communicate.save(mp3_path))
        converter = self.convert_to_wav or _convert_mp3_to_wav
        converter(mp3_path, output_path, self.sample_rate)
        mp3_path.unlink(missing_ok=True)


@dataclass(frozen=True)
class GeminiTts:
    api_key: str
    model: str = "gemini-2.5-flash-preview-tts"
    voice_name: str = "Kore"
    sample_rate: int = 24000
    duration_sec: float = 0.0
    engine_name: str = "gemini-tts"
    transport: GeminiTransport | None = None
    timeout_seconds: int = 60
    max_retries: int = 3
    sleep: Callable[[float], None] = time.sleep

    def synthesize(self, text: str, output_path: Path) -> None:
        response = self._post_with_retry(self._build_payload(text))
        inline_data = _extract_audio_inline_data(response)
        pcm_bytes = base64.b64decode(inline_data["data"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_pcm_wav(output_path, pcm_bytes, sample_rate=self.sample_rate)

    def _post_with_retry(self, payload: dict[str, Any]) -> dict[str, Any]:
        transport = self.transport or UrllibGeminiTransport()
        for attempt in range(1, self.max_retries + 1):
            try:
                return transport.post_json(
                    self._generate_content_url(),
                    payload=payload,
                    timeout_seconds=self.timeout_seconds,
                )
            except RetryableGeminiError:
                if attempt == self.max_retries:
                    raise
                self.sleep(float(2 ** (attempt - 1)))
        raise RuntimeError("unreachable retry state")

    def _generate_content_url(self) -> str:
        return (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )

    def _build_payload(self, text: str) -> dict[str, Any]:
        return {
            "contents": [{"parts": [{"text": f"Say clearly in Indonesian: {text}"}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {"voiceName": self.voice_name}
                    }
                },
            },
        }


def _convert_mp3_to_wav(mp3_path: Path, wav_path: Path, sample_rate: int) -> None:
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(mp3_path),
                "-ac",
                "1",
                "-ar",
                str(sample_rate),
                str(wav_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    subprocess.run(
        [
            "afconvert",
            str(mp3_path),
            str(wav_path),
            "-f",
            "WAVE",
            "-d",
            f"LEI16@{sample_rate}",
            "-c",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _extract_audio_inline_data(response: dict[str, Any]) -> dict[str, str]:
    parts = response["candidates"][0]["content"]["parts"]
    for part in parts:
        inline_data = part.get("inlineData")
        if inline_data:
            return inline_data
    raise ValueError("Gemini TTS response missing inline audio data")


def _write_pcm_wav(output_path: Path, pcm_bytes: bytes, sample_rate: int) -> None:
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)


def _read_wav_duration_sec(path: Path) -> float:
    with wave.open(str(path), "rb") as wav_file:
        return wav_file.getnframes() / wav_file.getframerate()


def build_audio_manifest_rows(
    rows: Iterable[dict],
    audio_dir: Path,
    tts: TtsEngine,
    processed_utterance_ids: set[str] | None = None,
    rate_limiter: RateLimiter | None = None,
) -> list[dict]:
    audio_rows = []
    processed_utterance_ids = processed_utterance_ids or set()
    for row in rows:
        utterance_id = row["utterance_id"]
        if utterance_id in processed_utterance_ids:
            continue
        text = row.get("text")
        if not text:
            raise ValueError(f"row {utterance_id} missing text")

        audio_path = audio_dir / f"{utterance_id}.wav"
        if rate_limiter is not None:
            rate_limiter.wait_before_request()
        tts.synthesize(text, audio_path)
        audio_rows.append(
            {
                **row,
                "audio_path": str(audio_path),
                "duration_sec": _read_wav_duration_sec(audio_path),
                "sample_rate": tts.sample_rate,
                "tts_engine": tts.engine_name,
            }
        )

    return audio_rows
