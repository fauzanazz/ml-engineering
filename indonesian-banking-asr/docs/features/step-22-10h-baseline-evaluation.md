---
title: "Step 22: 10h Baseline Evaluation"
type: [feature-note, evaluation, baseline]
created: 2026-05-16
status: completed
categories: [asr, whisper, mlx, evaluation, baseline]
related:
  - step-21-10h-80-20-data-procurement.md
  - step-20-guarded-sgd-scaleup-evaluation.md
  - ../../README.md
author: fauzan
---

# Step 22: 10h Baseline Evaluation

Step ini menjalankan baseline MLX Whisper large-v3 pada validation/test dari kandidat dataset 10h 80/20. Tujuannya membuat baseline sebelum training adapter/LoRA atau eksperimen fine-tuning berikutnya.

## Evaluation Manifests

Built from `data/training/combined_manifest_10h_80_20_candidate.jsonl`.

| Manifest | Rows | Hours |
|---|---:|---:|
| `artifacts/combined_10h_validation_manifest.jsonl` | 485 | 0.9900 |
| `artifacts/combined_10h_test_manifest.jsonl` | 511 | 1.0654 |
| `artifacts/combined_10h_nontrain_manifest.jsonl` | 996 | 2.0553 |
| `artifacts/combined_10h_validation_real_manifest.jsonl` | 353 | 0.7675 |
| `artifacts/combined_10h_validation_banking_manifest.jsonl` | 132 | 0.2225 |
| `artifacts/combined_10h_test_real_manifest.jsonl` | 355 | 0.8059 |
| `artifacts/combined_10h_test_banking_manifest.jsonl` | 156 | 0.2595 |

## Baseline Transcription

Commands:

```bash
uv run banking-asr-transcribe-whisper \
  --manifest-path artifacts/combined_10h_validation_manifest.jsonl \
  --output-path artifacts/mlx_whisper_large_v3_baseline_combined_10h_validation_predictions.jsonl \
  --model mlx-community/whisper-large-v3-mlx
```

```bash
uv run banking-asr-transcribe-whisper \
  --manifest-path artifacts/combined_10h_test_manifest.jsonl \
  --output-path artifacts/mlx_whisper_large_v3_baseline_combined_10h_test_predictions.jsonl \
  --model mlx-community/whisper-large-v3-mlx
```

## Results

Raw baseline metrics:

| Eval slice | Rows | WER | Entity error rate | Entity errors |
|---|---:|---:|---:|---:|
| validation combined | 485 | 12.11% | 17.27% | 72 / 417 |
| validation real non-banking | 353 | 12.51% | 0.00% | 0 / 0 |
| validation banking synthetic | 132 | 9.69% | 17.27% | 72 / 417 |
| test combined | 511 | 13.02% | 13.96% | 67 / 480 |
| test real non-banking | 355 | 12.80% | 0.00% | 0 / 0 |
| test banking synthetic | 156 | 14.23% | 13.96% | 67 / 480 |

Validation banking entity errors:

```json
{"ACCOUNT_NUMBER": 15, "CARD_LAST4": 12, "PRODUCT_NAME": 18, "BANKING_TERM": 15, "TRANSFER_METHOD": 9, "MERCHANT_NAME": 3}
```

Test banking entity errors:

```json
{"ACCOUNT_NUMBER": 21, "AMOUNT": 13, "CARD_LAST4": 6, "PRODUCT_NAME": 12, "BANKING_TERM": 12, "TRANSFER_METHOD": 3}
```

## Interpretation

Whisper large-v3 remains strong on Indonesian general ASR: real non-banking WER is about 12.5-12.8% on the 10h eval slices. Banking WER is also reasonable, but entity error rate is still high at 14-17% without domain post-processing.

This confirms the next training/evaluation target should optimize banking entity preservation, not generic Indonesian recognition.

## Decision

Baseline for 10h candidate is established. Next productive step is one of:

```text
1. entity-aware post-processing evaluation on the full 10h validation/test slices
2. adapter/LoRA-style training with 10h train manifest
3. banking data quality pass to replace dry-run text with live paraphrases
```

Recommended Step 23: run entity-aware post-processing on these predictions first, because it is cheap and directly targets the largest current gap.
