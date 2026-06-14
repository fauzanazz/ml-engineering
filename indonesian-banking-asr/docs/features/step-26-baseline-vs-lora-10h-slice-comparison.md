---
title: "Step 26: Baseline vs LoRA 10h Slice Comparison"
type: [feature-note, evaluation, analysis]
created: 2026-05-17
status: completed
categories: [asr, whisper, mlx, lora, evaluation, error-analysis]
related:
  - step-25-10h-lora-scaleup-evaluation.md
  - step-24-mlx-whisper-lora-smoke.md
  - step-23-entity-aware-postprocessing-10h.md
author: fauzan
---

# Step 26: Baseline vs LoRA 10h Slice Comparison

Step ini membandingkan MLX Whisper large-v3 baseline dengan checkpoint LoRA 500-step dari Step 25 pada split 10h yang sama. Tujuannya memastikan apakah LoRA memberi perbaikan nyata sebelum menjalankan eksperimen 1000-step yang lebih mahal.

## Setup

Baseline artifacts sudah tersedia dari evaluasi Step 22/23. LoRA artifacts berasal dari Step 25.

Compared models:

| Model | Path |
|---|---|
| Baseline | `mlx-community/whisper-large-v3-mlx` |
| LoRA 500-step merged | `models/mlx-whisper-large-v3-10h-500step-lora-last4-r8-merged` |

Compared splits and slices:

| Split | Slice | Manifest |
|---|---|---|
| validation | all | `artifacts/combined_10h_validation_manifest.jsonl` |
| validation | synthetic banking | `artifacts/combined_10h_validation_banking_manifest.jsonl` |
| validation | non-banking real | `artifacts/combined_10h_validation_real_manifest.jsonl` |
| test | all | `artifacts/combined_10h_test_manifest.jsonl` |
| test | synthetic banking | `artifacts/combined_10h_test_banking_manifest.jsonl` |
| test | non-banking real | `artifacts/combined_10h_test_real_manifest.jsonl` |

## Commands

LoRA subset metrics were generated from Step 25 predictions without retranscribing audio:

```bash
uv run banking-asr-evaluate \
  --manifest-path artifacts/combined_10h_validation_banking_manifest.jsonl \
  --predictions-path artifacts/mlx_whisper_large_v3_10h_500step_lora_last4_r8_merged_validation_predictions.jsonl \
  --output-path artifacts/mlx_whisper_large_v3_10h_500step_lora_last4_r8_merged_validation_banking_synthetic_eval.jsonl

uv run banking-asr-evaluate \
  --manifest-path artifacts/combined_10h_validation_real_manifest.jsonl \
  --predictions-path artifacts/mlx_whisper_large_v3_10h_500step_lora_last4_r8_merged_validation_predictions.jsonl \
  --output-path artifacts/mlx_whisper_large_v3_10h_500step_lora_last4_r8_merged_validation_non_banking_real_eval.jsonl

uv run banking-asr-evaluate \
  --manifest-path artifacts/combined_10h_test_banking_manifest.jsonl \
  --predictions-path artifacts/mlx_whisper_large_v3_10h_500step_lora_last4_r8_merged_test_predictions.jsonl \
  --output-path artifacts/mlx_whisper_large_v3_10h_500step_lora_last4_r8_merged_test_banking_synthetic_eval.jsonl

uv run banking-asr-evaluate \
  --manifest-path artifacts/combined_10h_test_real_manifest.jsonl \
  --predictions-path artifacts/mlx_whisper_large_v3_10h_500step_lora_last4_r8_merged_test_predictions.jsonl \
  --output-path artifacts/mlx_whisper_large_v3_10h_500step_lora_last4_r8_merged_test_non_banking_real_eval.jsonl
```

Same command pattern was repeated for postprocessed predictions.

## Results

| Split | Slice | Mode | Rows | Baseline WER | LoRA WER | Baseline entity error | LoRA entity error |
|---|---|---|---:|---:|---:|---:|---:|
| validation | all | raw | 485 | 12.11% | 12.10% | 17.27% | 17.27% |
| validation | all | postprocessed | 485 | 11.33% | 11.32% | 1.44% | 1.44% |
| validation | synthetic banking | raw | 132 | 9.69% | 9.69% | 17.27% | 17.27% |
| validation | synthetic banking | postprocessed | 132 | 4.19% | 4.19% | 1.44% | 1.44% |
| validation | non-banking real | raw | 353 | 12.51% | 12.50% | 0.00% | 0.00% |
| validation | non-banking real | postprocessed | 353 | 12.51% | 12.50% | 0.00% | 0.00% |
| test | all | raw | 511 | 13.02% | 13.02% | 13.96% | 13.96% |
| test | all | postprocessed | 511 | 12.43% | 12.43% | 3.33% | 3.33% |
| test | synthetic banking | raw | 156 | 14.23% | 14.23% | 13.96% | 13.96% |
| test | synthetic banking | postprocessed | 156 | 10.30% | 10.30% | 3.33% | 3.33% |
| test | non-banking real | raw | 355 | 12.80% | 12.80% | 0.00% | 0.00% |
| test | non-banking real | postprocessed | 355 | 12.81% | 12.81% | 0.00% | 0.00% |

## Output Difference Check

Raw transcript differences were tiny:

| Split | Changed hypotheses | Rows |
|---|---:|---:|
| validation | 5 | 485 |
| test | 3 | 511 |

Most changes were punctuation, casing, or spacing. One validation example changed `9.20` to `920`, which can hurt numeric formatting even when semantic content is close.

## Decision

LoRA 500-step is stable but does not materially improve 10h validation/test quality over baseline. Entity-aware postprocessing remains the dominant improvement lever for banking phrases and account-like entities.

Do not spend compute on 1000-step LoRA with the same configuration yet. Next experiment should change one stronger variable first:

```text
train decoder_last_8_lora or increase LoRA rank, then evaluate the same 10h slices
```

