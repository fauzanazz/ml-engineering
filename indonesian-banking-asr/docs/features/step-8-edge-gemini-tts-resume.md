---
title: "Step 8: Edge TTS, Gemini TTS & Resumable TTS"
type: [feature-note, implementation]
created: 2026-05-15
status: completed
categories: [asr, synthetic-data, tts, edge-tts, gemini, resume]
related:
  - step-7-audio-manifest-qa-dataset-assembly.md
  - ../../README.md
author: fauzan
---

# Step 8: Edge TTS, Gemini TTS & Resumable TTS

Step ini menambahkan real speech TTS provider untuk mengganti placeholder tone, dengan fokus utama pada Edge TTS karena tidak bergantung pada Google Cloud credit dan cukup baik untuk pilot dataset.

## Spesifikasi

Tujuan minimal:

- Tambah provider TTS nyata untuk synthetic banking utterances.
- Support Gemini TTS untuk smoke/small sample saat quota tersedia.
- Support Edge TTS untuk main pilot dataset.
- Convert provider output MP3/PCM menjadi mono WAV.
- Tambah resume dan fixed delay untuk TTS batch kecil.

## Files Added / Updated

Updated:

```text
pyproject.toml
src/indonesian_banking_asr/synthetic/tts.py
src/indonesian_banking_asr/synthetic/cli.py
tests/synthetic/test_tts.py
tests/synthetic/test_cli.py
```

Dependency added:

```text
edge-tts>=7.2.8
```

## Provider: Edge TTS

Edge TTS is the main practical provider for this project.

Default Indonesian voice:

```text
id-ID-ArdiNeural
```

Alternative Indonesian voice:

```text
id-ID-GadisNeural
```

CLI example:

```bash
uv run banking-asr-generate-text tts \
  --provider edge \
  --voice id-ID-ArdiNeural \
  --sample-rate 16000 \
  --input-path data/synthetic/accepted_manifest.jsonl \
  --output-path data/synthetic/edge_audio_manifest.jsonl \
  --audio-dir data/synthetic/edge_audio_clean \
  --resume \
  --seconds-per-request 1
```

Implementation behavior:

```text
text -> edge-tts MP3 -> ffmpeg/afconvert -> mono WAV -> audio manifest
```

The converter tries `ffmpeg` first and falls back to macOS `afconvert`.

## Provider: Gemini TTS

Gemini TTS was added and verified, but it is not the main provider because the free-tier TTS quota is too small for dataset generation.

Verified model:

```text
gemini-2.5-flash-preview-tts
```

Observed free-tier limit:

```text
10 requests/day/project/model
```

CLI example for small smoke sample:

```bash
uv run banking-asr-generate-text tts \
  --provider gemini \
  --voice Kore \
  --input-path data/synthetic/accepted_one.jsonl \
  --output-path data/synthetic/gemini_audio_manifest_one.jsonl \
  --audio-dir data/synthetic/gemini_audio_clean_one \
  --resume
```

Gemini TTS returns raw PCM:

```text
audio/L16;codec=pcm;rate=24000
```

The implementation writes PCM bytes into WAV.

## TTS Resume & Delay

TTS subcommand supports:

```bash
--resume
--seconds-per-request 1
```

Resume behavior:

```text
read existing audio manifest
collect processed utterance_id
skip processed rows
keep existing rows
append newly generated rows
rewrite merged manifest
```

This prevents restarting TTS from zero after quota, network, or provider errors.

## Verification

Full test suite:

```bash
uv run pytest
```

Live Edge TTS smoke result:

```text
voice: id-ID-ArdiNeural
sample_rate: 16000
channels: 1
sampwidth: 2
duration_sec: 5.784
```

Gemini TTS access test passed once, then later hit quota:

```text
HTTP: OK
MIME_TYPE: audio/L16;codec=pcm;rate=24000
```

Quota error observed:

```text
Quota exceeded ... model: gemini-2.5-flash-tts ... quotaValue: 10
```

## Current Limitations

- Edge TTS is an online service; rate limits and service policy should be respected.
- Gemini TTS free-tier quota is too small for normal dataset generation.
- Resume trusts existing manifest rows; run audio QA after every resumed batch.

## Next Step

Step 9 should route TTS through 9Router so the same pipeline can use Edge TTS behind one HTTP endpoint.

## Related

- [Step 7: Audio Manifest, QA & Dataset Assembly](step-7-audio-manifest-qa-dataset-assembly.md)
