---
title: "Step 31: Banking-Only LoRA at lr=1e-4"
type: [feature-note, training, evaluation, hyperparameter]
created: 2026-05-17
status: completed
categories: [asr, whisper, mlx, lora, learning-rate]
related:
  - step-30-banking-only-lora-lr5e-5.md
  - step-29-banking-only-lora-200step.md
author: fauzan
---

# Step 31: Banking-Only LoRA at lr=1e-4

Step ini melanjutkan sweep learning rate dari Step 30 (5e-5) ke 1e-4 dengan konfigurasi identik, untuk melihat apakah perbaikan masih linier atau mulai jenuh.

## Training

Command:

```bash
uv run banking-asr-finetune-mlx-whisper \
  --manifest-path artifacts/synthetic_banking_train_manifest_10h_candidate.jsonl \
  --output-dir models/mlx-whisper-large-v3-banking-only-200step-lora-last8-r8-lr1e-4-merged \
  --summary-path artifacts/mlx_whisper_large_v3_banking_only_200step_lora_last8_r8_lr1e-4_merged_summary.jsonl \
  --model-name mlx-community/whisper-large-v3-mlx \
  --max-steps 200 \
  --train-scope decoder_last_8_lora \
  --optimizer adamw \
  --learning-rate 1e-4 \
  --warmup-steps 20 \
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
  "rows_seen": 1008,
  "completed_steps": 200,
  "learning_rate": 0.0001,
  "first_loss": 0.7034010291099548,
  "last_loss": 0.09318286180496216
}
```

## Evaluation

Validation banking 20-row subset, learning-rate sweep summary:

| Model | Raw WER | Raw entity error | Postprocessed WER | Postprocessed entity error |
|---|---:|---:|---:|---:|
| Baseline | 4.47% | 10.00% | 1.12% | 0.00% |
| LoRA lr=1e-6 (Step 29) | 4.47% | 10.00% | 1.12% | 0.00% |
| LoRA lr=5e-5 (Step 30) | 3.91% | 8.33% | 1.12% | 0.00% |
| LoRA lr=1e-4 (Step 31) | 2.79% | 5.00% | 1.12% | 0.00% |

Raw entity error halved versus baseline. Remaining raw errors are 1 ACCOUNT_NUMBER and 2 PRODUCT_NAME, no more CARD_LAST4 errors.

## Observations

- Learning rate is the dominant lever for this LoRA stack; quality improves monotonically from 1e-6 to 1e-4.
- Postprocessing still flattens to the same 1.12% WER / 0% entity error, so the model gain is currently masked after postprocessing on this small subset.
- The interesting open question is whether raw gains survive on the full 10h validation/test and on non-banking real audio (regression risk).

## Decision

Promote lr=1e-4 banking-only last-8 LoRA to a full-split evaluation to measure real WER/entity movement and any non-banking regression before pushing lr higher or steps longer.

Next action:

```text
transcribe and evaluate lr=1e-4 model on combined_10h validation/test all + banking + real slices, raw and postprocessed
```

