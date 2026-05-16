---
title: "Step 10: Whisper Baseline & Banking Post-processing"
type: [feature-note, evaluation]
created: 2026-05-16
status: completed
categories: [asr, evaluation, whisper, mlx, post-processing, banking]
related:
  - step-9-9router-edge-tts-scaling.md
  - ../../README.md
author: fauzan
---

# Step 10: Whisper Baseline & Banking Post-processing

Step ini membuat baseline Whisper large-v3 untuk dataset audio 9Router Edge TTS dan menambahkan post-processing domain banking untuk memperbaiki entity formatting.

## Spesifikasi

Tujuan minimal:

- Jalankan baseline [Whisper large-v3](https://huggingface.co/mlx-community/whisper-large-v3-mlx) lokal di Apple Silicon memakai `mlx-whisper`.
- Evaluasi 150 clean WAV dari [Step 9](step-9-9router-edge-tts-scaling.md).
- Bandingkan raw transcript dan post-processed transcript.
- Fokus metrik utama pada WER dan entity error rate.

## Files Added / Updated

Added:

```text
src/indonesian_banking_asr/evaluation/postprocess.py
tests/evaluation/test_postprocess.py
```

Generated artifacts under ignored `data/synthetic/`:

```text
data/synthetic/mlx_whisper_large_v3_predictions_150.jsonl
data/synthetic/mlx_whisper_large_v3_eval_summary_150.jsonl
data/synthetic/mlx_whisper_large_v3_postprocessed_predictions_150.jsonl
data/synthetic/mlx_whisper_large_v3_postprocessed_eval_summary_150.jsonl
data/synthetic/mlx_whisper_large_v3_error_report_150.jsonl
```

## Baseline Model

Apple Silicon path uses MLX instead of faster-whisper because faster-whisper is CPU-only on this Mac.

```text
model: mlx-community/whisper-large-v3-mlx
runtime: mlx-whisper
language: id
input: data/synthetic/9router_edge_audio_manifest_50.jsonl
rows: 150 clean WAV files
```

The model was downloaded with Hugging Face CLI:

```bash
hf download mlx-community/whisper-large-v3-mlx
```

## Raw Whisper Result

Raw `mlx-whisper` output on 150 clean rows:

```json
{
  "rows": 150,
  "word_errors": 237,
  "reference_words": 1508,
  "wer": 0.15716180371352786,
  "entity_errors": 61,
  "entities": 465,
  "entity_error_rate": 0.13118279569892474,
  "errors_by_type": {
    "PRODUCT_NAME": 22,
    "ACCOUNT_NUMBER": 11,
    "AMOUNT": 6,
    "BANKING_TERM": 15,
    "CARD_LAST4": 1,
    "TRANSFER_METHOD": 3,
    "MERCHANT_NAME": 3
  }
}
```

Summary:

```text
Raw WER: 15.72%
Raw entity error rate: 13.12%
```

## Post-processing Rules

Post-processing is intentionally small and domain-specific. It fixes transcript forms that are acoustically plausible but fail exact entity matching.

Rules include:

```text
0 4 3 3 2 1 8 1 9 6 -> 0433218196
0 433 218 196 -> 0433218196
0433-218-196 -> 0433218196
2751-79790869-nya -> 275179790869-nya
9.027 -> 9027
17.110.000 rupiah -> Rp17.110.000
RP36010000 -> Rp36.010.000
RP36-010-000 -> Rp36.010.000
Rp 17.110.000 -> Rp17.110.000
BIFAS / BI Fast / Beifast / bevas -> BI-FAST
Kris / Keris / Krisaya / Chris815... -> QRIS
bloket -> blocked
pilater / pelater -> paylater
Sopiko / Sopi -> Shopee
```

## Post-processed Whisper Result

After post-processing:

```json
{
  "rows": 150,
  "word_errors": 155,
  "reference_words": 1508,
  "wer": 0.10278514588859416,
  "entity_errors": 0,
  "entities": 465,
  "entity_error_rate": 0.0,
  "errors_by_type": {}
}
```

Summary:

```text
Post-processed WER: 10.28%
Post-processed entity error rate: 0.00%
```

## AssemblyAI Sanity Check

AssemblyAI was used only as a listening/STT sanity proxy, not as the project baseline. The direct API request used:

```json
{
  "speech_models": ["universal-3-pro", "universal-2"],
  "language_detection": true
}
```

Apple-to-apple comparison on the 126 rows that AssemblyAI completed:

```text
AssemblyAI raw WER: 21.52%
AssemblyAI entity error rate: 18.75%
MLX Whisper large-v3 raw WER on same rows: 16.56%
MLX Whisper large-v3 entity error rate on same rows: 14.06%
```

This confirmed Whisper large-v3 is the stronger baseline for this synthetic Indonesian banking dataset.

## Verification

Automated tests:

```bash
uv run pytest
```

Observed result:

```text
77 passed
```

Audio/data QA from Step 9 remained valid:

```json
{"checked_rows": 150, "valid_rows": 150, "invalid_rows": 0, "errors": []}
```

Dataset summary:

```json
{
  "total_rows": 450,
  "dataset_variant_counts": {"clean": 150, "augmented": 300},
  "split_counts": {"train": 342, "validation": 27, "test": 81},
  "augmentation_profile_counts": {
    "call_low_noise": 150,
    "call_medium_noise": 150
  }
}
```

## Current Limitations

- Post-processing is tuned on the 150-row pilot, so it needs validation on larger and more varied rows.
- Some replacements such as `Kris`/`Keris` to `QRIS` are banking-domain assumptions and may be unsafe in open-domain ASR.
- Evaluation still uses strict exact entity substring matching after basic normalization.
- Augmented audio rows have not yet been transcribed with Whisper.

## Next Step

Step 11 should run the same Whisper + post-processing evaluation on the full 450-row clean+augmented manifest, then separate metrics by `dataset_variant` and `augmentation.name`.

## Related

- [Step 9: 9Router Edge TTS & Generation Scaling](step-9-9router-edge-tts-scaling.md)
- [Project README](../../README.md)
