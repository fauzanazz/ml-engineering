---
title: "Step 9: 9Router Edge TTS & Generation Scaling"
type: [feature-note, implementation]
created: 2026-05-15
status: completed
categories: [asr, synthetic-data, tts, 9router, edge-tts, scaling]
related:
  - step-8-edge-gemini-tts-resume.md
  - ../../README.md
author: fauzan
---

# Step 9: 9Router Edge TTS & Generation Scaling

Step ini menambahkan 9Router TTS sebagai HTTP wrapper untuk Edge TTS dan menambahkan `--samples-per-template` agar text generation bisa melewati batas 12 template awal.

## Spesifikasi

Tujuan minimal:

- Tambah provider `9router` di TTS CLI.
- Gunakan Edge TTS voice melalui endpoint 9Router.
- Support `NINEROUTER_URL` dan optional `NINEROUTER_KEY`.
- Convert MP3 response dari 9Router menjadi WAV.
- Tambah `--samples-per-template` untuk canonical text scaling.

## Files Added / Updated

Updated:

```text
src/indonesian_banking_asr/synthetic/tts.py
src/indonesian_banking_asr/synthetic/cli.py
src/indonesian_banking_asr/synthetic/pipeline.py
tests/synthetic/test_tts.py
tests/synthetic/test_pipeline.py
```

## 9Router TTS Provider

Environment:

```bash
NINEROUTER_URL=...
NINEROUTER_KEY=... # optional if server auth enabled
```

Provider:

```bash
--provider 9router
```

Main voice:

```text
edge-tts/id-ID-ArdiNeural
```

Alternative voice:

```text
edge-tts/id-ID-GadisNeural
```

CLI example:

```bash
set -a
. ./.env
set +a

uv run banking-asr-generate-text tts \
  --provider 9router \
  --voice edge-tts/id-ID-ArdiNeural \
  --sample-rate 16000 \
  --input-path data/synthetic/accepted_manifest.jsonl \
  --output-path data/synthetic/9router_edge_audio_manifest.jsonl \
  --audio-dir data/synthetic/9router_edge_audio_clean \
  --resume \
  --seconds-per-request 1
```

Implementation behavior:

```text
POST $NINEROUTER_URL/v1/audio/speech
body: {"model": "edge-tts/id-ID-ArdiNeural", "input": text}
response: audio/mp3 bytes
convert: MP3 -> mono WAV
manifest: audio_path, duration_sec, sample_rate, tts_engine
```

## Voice Discovery

Observed Indonesian voices:

```text
edge-tts/id-ID-ArdiNeural
edge-tts/id-ID-GadisNeural
```

Discovery command:

```bash
curl "$NINEROUTER_URL/v1/audio/voices?provider=edge-tts&lang=id"
```

## Scaling: samples-per-template

Before this step, generation was bounded by catalog size:

```text
12 templates -> max 12 canonical rows
```

New CLI flag:

```bash
--samples-per-template 5
```

Example:

```bash
uv run banking-asr-generate-text \
  --output-path data/synthetic/text_manifest_50_samples.jsonl \
  --seed 42 \
  --limit 50 \
  --samples-per-template 5
```

Observed result:

```text
rows 50
unique_templates 10
max_sample_index 4
syn_id_check_balance_001_000001_s00_p00
syn_id_check_balance_001_000002_s01_p00
```

Utterance IDs now include sample index:

```text
syn_id_<template_id>_<row_index>_s<sample_index>_p00
```

Rows include:

```text
template_sample_index
```

## Pilot Verification

9Router Edge TTS smoke:

```text
HTTP: OK
CONTENT_TYPE: audio/mp3
framerate 16000
channels 1
duration_sec 2.556
```

Five-row 9Router Edge flow:

```text
audio_manifest rows: 5
augmented_manifest rows: 10
dataset_manifest rows: 15
```

Audio QA:

```json
{"checked_rows": 5, "valid_rows": 5, "invalid_rows": 0, "errors": []}
```

Dataset QA:

```json
{"checked_rows": 15, "valid_rows": 15, "invalid_rows": 0, "errors": []}
```

Dataset summary:

```json
{
  "total_rows": 15,
  "dataset_variant_counts": {"clean": 5, "augmented": 10},
  "split_counts": {"train": 15},
  "augmentation_profile_counts": {
    "call_low_noise": 5,
    "call_medium_noise": 5
  }
}
```

## Current Limitations

- 9Router availability depends on local/server setup.
- `NINEROUTER_KEY` is optional only if the server is configured without auth.
- Scaling text rows still depends on template diversity; more templates are needed for better linguistic coverage.
- Audio quality still needs human review before training.

## Next Step

Step 10 should produce a reviewed pilot dataset using 50 canonical rows, Edge TTS through 9Router, QA reports, and a sample listening checklist.

## Related

- [Step 8: Edge TTS, Gemini TTS & Resumable TTS](step-8-edge-gemini-tts-resume.md)
