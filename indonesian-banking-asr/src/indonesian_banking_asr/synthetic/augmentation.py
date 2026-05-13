from __future__ import annotations

import random
import wave
from pathlib import Path
from typing import Iterable


def build_augmented_manifest_rows(
    rows: Iterable[dict],
    output_dir: Path,
    gain: float,
    noise_amplitude: int = 0,
    seed: int = 42,
) -> list[dict]:
    if gain <= 0:
        raise ValueError("gain must be positive")
    if noise_amplitude < 0:
        raise ValueError("noise_amplitude cannot be negative")

    augmented_rows = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        utterance_id = row["utterance_id"]
        source_audio_path = Path(row["audio_path"])
        augmented_audio_path = output_dir / f"{utterance_id}_gain-{gain}.wav"
        _write_augmented_wav(source_audio_path, augmented_audio_path, gain, noise_amplitude, seed)
        augmentation = {"gain": gain}
        if noise_amplitude:
            augmentation["noise_amplitude"] = noise_amplitude
            augmentation["seed"] = seed
        augmented_rows.append(
            {
                **row,
                "audio_path": str(augmented_audio_path),
                "source_audio_path": str(source_audio_path),
                "augmentation": augmentation,
            }
        )

    return augmented_rows


def _write_augmented_wav(
    source_path: Path,
    output_path: Path,
    gain: float,
    noise_amplitude: int,
    seed: int,
) -> None:
    with wave.open(str(source_path), "rb") as source_wav:
        params = source_wav.getparams()
        frames = source_wav.readframes(source_wav.getnframes())

    if params.sampwidth != 2:
        raise ValueError("only 16-bit PCM wav is supported")

    noise = random.Random(seed)
    scaled_frames = bytearray()
    for index in range(0, len(frames), 2):
        sample = int.from_bytes(frames[index : index + 2], byteorder="little", signed=True)
        noise_sample = noise.randint(-noise_amplitude, noise_amplitude) if noise_amplitude else 0
        scaled_sample = max(-32768, min(32767, int(sample * gain) + noise_sample))
        scaled_frames.extend(scaled_sample.to_bytes(2, byteorder="little", signed=True))

    with wave.open(str(output_path), "wb") as output_wav:
        output_wav.setparams(params)
        output_wav.writeframes(bytes(scaled_frames))
