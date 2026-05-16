---
title: "Step 12: BabelSpeech Real-data Grounding"
type: [feature-note, data, evaluation]
created: 2026-05-16
status: completed
categories: [asr, evaluation, whisper, real-data, babelspeech]
related:
  - step-11-augmented-whisper-evaluation.md
  - ../../README.md
author: fauzan
---

# Step 12: BabelSpeech Real-data Grounding

Step ini menambahkan jalur real-data untuk mengecek apakah evaluasi synthetic banking sudah masuk akal terhadap distribusi Indonesian colloquial speech publik.

## Dataset Schema

Dataset inspected:

```text
BabelSpeech/40hours_Indonesian_Colloquial_ASR_Speech_Dataset
/Users/fauzan/.cache/huggingface/hub/datasets--BabelSpeech--40hours_Indonesian_Colloquial_ASR_Speech_Dataset/snapshots/7af54dbe9e40a888bc9fc92a183308e7ee78f14a
```

Files:

```text
README.md
audio_info.json
wav.zip
```

Metadata fields observed in `audio_info.json`:

```json
{
  "filename": "id_corpus_id_spk009_eGzAtizmL74_202509_0010_02.wav",
  "relative_path": "wav/id_corpus_id_spk009_eGzAtizmL74_202509_0010_02.wav",
  "duration": 6.1,
  "confidence": 1.0,
  "text": "bahkan Jembatan Lima dan kalau sore itu pada datang main ke pantai. Nah, dulunya Pulau Galang ini",
  "snr": 57.56049728393555,
  "dnsmos": 4.4216657
}
```

## Implementation

Added project-format converter:

```text
src/indonesian_banking_asr/real_data/babelspeech.py
tests/real_data/test_babelspeech.py
```

Output manifest fields:

```text
utterance_id, audio_path, text, split, source
```

The converter also preserves quality metadata for analysis:

```text
duration, confidence, snr, dnsmos
```

Added MLX Whisper manifest runner:

```text
src/indonesian_banking_asr/evaluation/whisper.py
tests/evaluation/test_whisper.py
```

CLI entry points:

```text
banking-asr-convert-babelspeech
banking-asr-transcribe-whisper
```

## Commands Run

Convert 50-row sample and extract corresponding WAV files:

```bash
uv run banking-asr-convert-babelspeech \
  --dataset-root /Users/fauzan/.cache/huggingface/hub/datasets--BabelSpeech--40hours_Indonesian_Colloquial_ASR_Speech_Dataset/snapshots/7af54dbe9e40a888bc9fc92a183308e7ee78f14a \
  --output-path data/real/babelspeech_manifest_50.jsonl \
  --limit 50 \
  --extract-audio
```

Transcribe with MLX Whisper large-v3:

```bash
uv run banking-asr-transcribe-whisper \
  --manifest-path data/real/babelspeech_manifest_50.jsonl \
  --output-path data/real/mlx_whisper_large_v3_predictions_50.jsonl \
  --model mlx-community/whisper-large-v3-mlx \
  --language id
```

Evaluate:

```bash
uv run banking-asr-evaluate \
  --manifest-path data/real/babelspeech_manifest_50.jsonl \
  --predictions-path data/real/mlx_whisper_large_v3_predictions_50.jsonl \
  --output-path data/real/mlx_whisper_large_v3_eval_summary_50.jsonl
```

Tests:

```bash
uv run pytest tests/real_data/test_babelspeech.py tests/evaluation/test_whisper.py
```

Result:

```text
4 passed
```

## BabelSpeech Sample Result

Sample manifest:

```text
rows: 50
split: train 40, validation 5, test 5
duration: 369.1 seconds
```

Raw MLX Whisper large-v3 on 50 real colloquial rows:

```json
{
  "rows": 50,
  "word_errors": 122,
  "reference_words": 1023,
  "wer": 0.11925708699902249,
  "entity_errors": 0,
  "entities": 0,
  "entity_error_rate": 0.0
}
```

Summary:

```text
Real colloquial BabelSpeech sample WER: 11.93%
Synthetic banking clean raw WER: 15.72%
Synthetic banking clean post-processed WER: 10.28%
Synthetic banking clean+aug raw WER: 15.76%
Synthetic banking clean+aug post-processed WER: 10.30%
```

## Interpretation

BabelSpeech 50-row sample is broad colloquial Indonesian and has no labeled banking entities, so `entity_error_rate` is not meaningful there. It is useful for general Indonesian ASR grounding: MLX Whisper large-v3 reaches 11.93% strict WER on the sample, close to the post-processed synthetic banking WER around 10.3% and better than raw synthetic banking WER around 15.7%.

Synthetic banking remains harder for entity formatting before post-processing because it contains account numbers, money amounts, product terms, QRIS/card terms, and artificial support-call wording. BabelSpeech is better as distribution coverage, not as direct banking metric replacement.

## Training Mix Plan

Recommended role split:

| Dataset | Role | Use |
|---|---|---|
| BabelSpeech | General Indonesian distribution | Build robustness for colloquial phrasing, speakers, everyday domains, and real acoustic variation. |
| Synthetic banking clean | Domain/entity adaptation | Teach banking vocabulary, number formats, product names, and entity-preserving transcripts. |
| Synthetic banking augmented | Call-channel adaptation | Teach noisy call-like audio, lower gain, and degradation profiles while preserving entity labels. |

Initial fine-tuning mix proposal:

```text
70% BabelSpeech general Indonesian
20% synthetic banking clean
10% synthetic banking augmented
```

Evaluation gates:

- Keep held-out BabelSpeech test rows to detect general Indonesian regression.
- Keep synthetic banking test rows for `AMOUNT`, `ACCOUNT_NUMBER`, product, and QRIS/card entity accuracy.
- Report both standard WER and entity error rate; do not use BabelSpeech entity rate because it has no entity labels.
- Compare raw model output separately from banking post-processing so training gains are not hidden by rules.

## Current Limitations

- The run used a 50-row BabelSpeech sample, not the full 40/50-hour dataset.
- WER is strict whitespace-token WER without punctuation/case normalization.
- Extracted WAV and generated predictions live under ignored `data/real/`.
- Full BabelSpeech transcription is now wired but should be run when runtime and storage budget are acceptable.

## Next Step

Run full BabelSpeech conversion/transcription, then tune a normalized WER metric before any model fine-tuning decision.
