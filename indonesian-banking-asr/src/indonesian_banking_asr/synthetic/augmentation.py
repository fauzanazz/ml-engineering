from __future__ import annotations

import random
import wave
from pathlib import Path
from typing import Iterable


def build_augmented_manifest_rows(
    rows: Iterable[dict],
    output_dir: Path,
    gain: float | None = None,
    noise_amplitude: int = 0,
    seed: int = 42,
    profiles: list[dict] | None = None,
) -> list[dict]:
    if profiles is None:
        if gain is None:
            raise ValueError("gain is required")
        profiles = [{"gain": gain, "noise_amplitude": noise_amplitude, "seed": seed}]

    for profile in profiles:
        _validate_profile(profile)

    augmented_rows = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        for profile in profiles:
            augmented_rows.append(_build_augmented_row(row, output_dir, profile))

    return augmented_rows


def _validate_profile(profile: dict) -> None:
    if profile["gain"] <= 0:
        raise ValueError("gain must be positive")
    if profile.get("noise_amplitude", 0) < 0:
        raise ValueError("noise_amplitude cannot be negative")


def _build_augmented_row(row: dict, output_dir: Path, profile: dict) -> dict:
    utterance_id = row["utterance_id"]
    source_audio_path = Path(row["audio_path"])
    gain = profile["gain"]
    noise_amplitude = profile.get("noise_amplitude", 0)
    seed = profile.get("seed", 42)
    profile_name = profile.get("name")

    if profile_name is None:
        augmented_utterance_id = utterance_id
        augmented_audio_path = output_dir / f"{utterance_id}_gain-{gain}.wav"
        augmentation = {"gain": gain}
        if noise_amplitude:
            augmentation["noise_amplitude"] = noise_amplitude
            augmentation["seed"] = seed
    else:
        augmented_utterance_id = f"{utterance_id}_aug_{profile_name}"
        augmented_audio_path = output_dir / f"{augmented_utterance_id}.wav"
        augmentation = {
            "name": profile_name,
            "gain": gain,
            "noise_amplitude": noise_amplitude,
            "seed": seed,
        }

    _write_augmented_wav(source_audio_path, augmented_audio_path, gain, noise_amplitude, seed)
    return {
        **row,
        "utterance_id": augmented_utterance_id,
        "audio_path": str(augmented_audio_path),
        "source_audio_path": str(source_audio_path),
        "augmentation": augmentation,
    }


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
