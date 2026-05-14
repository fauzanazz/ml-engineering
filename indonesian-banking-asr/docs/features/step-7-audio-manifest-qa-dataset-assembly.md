---
title: "Step 7: Audio Manifest, QA & Dataset Assembly"
type: [feature-note, implementation]
created: 2026-05-15
status: completed
categories: [asr, synthetic-data, audio, qa, dataset]
related:
  - step-6-rate-limiter-resume-summary.md
  - ../../README.md
author: fauzan
---

# Step 7: Audio Manifest, QA & Dataset Assembly

Step ini mengubah text manifest menjadi dataset ASR audio lengkap: clean TTS manifest, audio QA, augmented audio manifest, merged dataset manifest, dataset QA, dan dataset summary.

## Spesifikasi

Tujuan minimal:

- Generate `.wav` per utterance dari text manifest.
- Simpan metadata audio di JSONL manifest.
- Validasi file WAV, sample rate, durasi, dan silent audio.
- Generate augmented audio dari clean audio.
- Merge clean + augmented manifests menjadi final dataset manifest.
- Tulis dataset QA dan summary JSONL.

## Files Added

```text
src/indonesian_banking_asr/synthetic/tts.py
src/indonesian_banking_asr/synthetic/audio_qa.py
src/indonesian_banking_asr/synthetic/augmentation.py
src/indonesian_banking_asr/synthetic/dataset.py
tests/synthetic/test_tts.py
tests/synthetic/test_audio_qa.py
tests/synthetic/test_augmentation.py
tests/synthetic/test_dataset.py
```

Updated:

```text
src/indonesian_banking_asr/synthetic/cli.py
```

## TTS Audio Manifest

Subcommand:

```bash
uv run banking-asr-generate-text tts \
  --input-path data/synthetic/accepted_manifest.jsonl \
  --output-path data/synthetic/audio_manifest.jsonl \
  --audio-dir data/synthetic/audio_clean \
  --sample-rate 8000 \
  --duration-sec 1.0
```

Initial implementation used `SyntheticToneTts` as a deterministic placeholder. It writes valid mono WAV files so the rest of the audio pipeline can be developed before real TTS is available.

Manifest fields added:

```text
audio_path
duration_sec
sample_rate
tts_engine
```

## Audio QA

Subcommand:

```bash
uv run banking-asr-generate-text audio-qa \
  --input-path data/synthetic/audio_manifest.jsonl \
  --output-path data/synthetic/audio_qa.jsonl
```

Checks:

```text
audio_path exists
valid wav file
not silent
sample_rate matches manifest
duration_sec matches manifest within one audio frame
```

Duration comparison uses one-frame tolerance because real TTS + MP3-to-WAV conversion can produce exact durations such as `5.784` while rounded QA would report `5.78`.

## Audio Augmentation

Subcommand:

```bash
uv run banking-asr-generate-text augment-audio \
  --input-path data/synthetic/audio_manifest.jsonl \
  --output-path data/synthetic/augmented_manifest.jsonl \
  --output-dir data/synthetic/audio_augmented \
  --profile call_low_noise:0.85:120:42 \
  --profile call_medium_noise:0.75:260:43
```

Profile format:

```text
name:gain:noise_amplitude:seed
```

Augmented rows include:

```text
source_audio_path
augmentation
dataset_variant=augmented
augmentation_profile
```

## Merge, QA & Summary

Merge clean and augmented manifests:

```bash
uv run banking-asr-generate-text merge-audio-manifests \
  --clean-input-path data/synthetic/audio_manifest.jsonl \
  --augmented-input-path data/synthetic/augmented_manifest.jsonl \
  --output-path data/synthetic/dataset_manifest.jsonl
```

Validate dataset manifest:

```bash
uv run banking-asr-generate-text dataset-qa \
  --input-path data/synthetic/dataset_manifest.jsonl \
  --output-path data/synthetic/dataset_qa.jsonl
```

Summarize dataset manifest:

```bash
uv run banking-asr-generate-text dataset-summary \
  --input-path data/synthetic/dataset_manifest.jsonl \
  --output-path data/synthetic/dataset_summary.jsonl
```

Summary fields:

```text
total_rows
dataset_variant_counts
split_counts
augmentation_profile_counts
```

## Verification

Unit tests:

```bash
uv run pytest
```

Representative smoke result from the one-row Edge/9Router pipeline:

```json
{"checked_rows": 1, "valid_rows": 1, "invalid_rows": 0, "errors": []}
```

Dataset summary example:

```json
{
  "total_rows": 2,
  "dataset_variant_counts": {"clean": 1, "augmented": 1},
  "split_counts": {"train": 2},
  "augmentation_profile_counts": {"call_low_noise": 1}
}
```

## Current Limitations

- `SyntheticToneTts` is valid audio but not speech.
- Augmentation is simple gain + additive noise, not full telco simulation yet.
- Dataset QA validates schema-level consistency, not transcription quality.

## Next Step

Step 8 should add real TTS providers and resumable TTS generation.

## Related

- [Step 6: Rate Limiter, Resume & Batch Summary](step-6-rate-limiter-resume-summary.md)
