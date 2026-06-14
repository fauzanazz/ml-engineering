---
title: "Step 25: 10h LoRA Scale-Up Evaluation"
type: [feature-note, training, evaluation]
created: 2026-05-16
status: completed
categories: [asr, whisper, mlx, lora, fine-tuning, evaluation]
related:
  - step-24-mlx-whisper-lora-smoke.md
  - step-23-entity-aware-postprocessing-10h.md
  - step-22-10h-baseline-evaluation.md
author: fauzan
---

# Step 25: 10h LoRA Scale-Up Evaluation

Step ini menjalankan scale-up dari LoRA path Step 24 pada full 10h training manifest, lalu mengevaluasi validation dan test split dengan raw transcript dan entity-aware postprocessing.

## Training

Command:

```bash
uv run banking-asr-finetune-mlx-whisper \
  --manifest-path data/training/combined_train_manifest_10h_80_20_candidate.jsonl \
  --output-dir models/mlx-whisper-large-v3-10h-500step-lora-last4-r8-merged \
  --summary-path artifacts/mlx_whisper_large_v3_10h_500step_lora_last4_r8_merged_summary.jsonl \
  --model-name mlx-community/whisper-large-v3-mlx \
  --max-steps 500 \
  --train-scope decoder_last_4_lora \
  --optimizer adamw \
  --learning-rate 1e-6 \
  --warmup-steps 50 \
  --lr-schedule warmup_linear_decay \
  --max-grad-norm 1.0 \
  --weight-decay 0.0 \
  --adam-beta2 0.98 \
  --adam-eps 1e-6 \
  --lora-rank 8 \
  --lora-alpha 16
```

Summary:

```json
{
  "rows_seen": 3839,
  "completed_steps": 500,
  "learning_rate": 1e-06,
  "warmup_steps": 50,
  "lr_schedule": "warmup_linear_decay",
  "optimizer": "adamw",
  "train_scope": "decoder_last_4_lora",
  "lora_rank": 8,
  "lora_alpha": 16.0,
  "lora_modules": 16,
  "first_loss": 0.788602888584137,
  "last_loss": 0.1895001381635666
}
```

Loss decreased cleanly across the 500-step run.

## Evaluation Manifests

Generated from `data/training/combined_manifest_10h_80_20_candidate.jsonl`:

| Split | Rows | Manifest |
|---|---:|---|
| validation | 485 | `artifacts/combined_10h_validation_manifest.jsonl` |
| test | 511 | `artifacts/combined_10h_test_manifest.jsonl` |

## Results

| Split | Mode | Rows | WER | Entity error rate |
|---|---|---:|---:|---:|
| validation | raw | 485 | 12.10% | 17.27% |
| validation | postprocessed | 485 | 11.32% | 1.44% |
| test | raw | 511 | 13.02% | 13.96% |
| test | postprocessed | 511 | 12.43% | 3.33% |

Artifacts:

```text
artifacts/mlx_whisper_large_v3_10h_500step_lora_last4_r8_merged_validation_predictions.jsonl
artifacts/mlx_whisper_large_v3_10h_500step_lora_last4_r8_merged_validation_eval.jsonl
artifacts/mlx_whisper_large_v3_10h_500step_lora_last4_r8_merged_validation_postprocessed_predictions.jsonl
artifacts/mlx_whisper_large_v3_10h_500step_lora_last4_r8_merged_validation_postprocessed_eval.jsonl
artifacts/mlx_whisper_large_v3_10h_500step_lora_last4_r8_merged_test_predictions.jsonl
artifacts/mlx_whisper_large_v3_10h_500step_lora_last4_r8_merged_test_eval.jsonl
artifacts/mlx_whisper_large_v3_10h_500step_lora_last4_r8_merged_test_postprocessed_predictions.jsonl
artifacts/mlx_whisper_large_v3_10h_500step_lora_last4_r8_merged_test_postprocessed_eval.jsonl
```

## Decision

The 500-step LoRA scale-up is stable and loadable. Postprocessing remains essential for entity quality, reducing validation entity error rate from 17.27% to 1.44% and test entity error rate from 13.96% to 3.33%.

Recommended next step:

```text
compare against baseline large-v3 on same combined validation/test manifests, then try 1000-step LoRA if WER improves or stays neutral
```
