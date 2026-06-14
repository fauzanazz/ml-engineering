---
title: "Step 28: Last-8 LoRA 500-Step Scale-Up"
type: [feature-note, training, evaluation]
created: 2026-05-17
status: completed
categories: [asr, whisper, mlx, lora, scale-up]
related:
  - step-27-last8-lora-smoke.md
  - step-26-baseline-vs-lora-10h-slice-comparison.md
  - step-25-10h-lora-scaleup-evaluation.md
author: fauzan
---

# Step 28: Last-8 LoRA 500-Step Scale-Up

Step ini menjalankan scale-up konfigurasi Step 27: LoRA pada 8 decoder block terakhir selama 500 step. Tujuannya menguji apakah kapasitas LoRA yang lebih luas memberi perubahan output yang tidak terlihat pada last-4 LoRA 500-step.

## Training

Command:

```bash
uv run banking-asr-finetune-mlx-whisper \
  --manifest-path data/training/combined_train_manifest_10h_80_20_candidate.jsonl \
  --output-dir models/mlx-whisper-large-v3-10h-500step-lora-last8-r8-merged \
  --summary-path artifacts/mlx_whisper_large_v3_10h_500step_lora_last8_r8_merged_summary.jsonl \
  --model-name mlx-community/whisper-large-v3-mlx \
  --max-steps 500 \
  --train-scope decoder_last_8_lora \
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
  "train_scope": "decoder_last_8_lora",
  "lora_rank": 8,
  "lora_alpha": 16.0,
  "lora_modules": 32,
  "first_loss": 0.788602888584137,
  "last_loss": 0.1894352287054062
}
```

Loss decreased cleanly and nearly matches last-4 LoRA 500-step final loss from Step 25.

## Smoke Evaluation

Validation subset:

```text
artifacts/combined_10h_validation_banking_20_manifest.jsonl
```

Commands:

```bash
uv run banking-asr-transcribe-whisper \
  --manifest-path artifacts/combined_10h_validation_banking_20_manifest.jsonl \
  --model models/mlx-whisper-large-v3-10h-500step-lora-last8-r8-merged \
  --output-path artifacts/mlx_whisper_large_v3_10h_500step_lora_last8_r8_merged_validation_banking_20_predictions.jsonl \
  --language id

uv run banking-asr-evaluate \
  --manifest-path artifacts/combined_10h_validation_banking_20_manifest.jsonl \
  --predictions-path artifacts/mlx_whisper_large_v3_10h_500step_lora_last8_r8_merged_validation_banking_20_predictions.jsonl \
  --output-path artifacts/mlx_whisper_large_v3_10h_500step_lora_last8_r8_merged_validation_banking_20_eval.jsonl

uv run banking-asr-postprocess-predictions \
  --predictions-path artifacts/mlx_whisper_large_v3_10h_500step_lora_last8_r8_merged_validation_banking_20_predictions.jsonl \
  --output-path artifacts/mlx_whisper_large_v3_10h_500step_lora_last8_r8_merged_validation_banking_20_postprocessed_predictions.jsonl

uv run banking-asr-evaluate \
  --manifest-path artifacts/combined_10h_validation_banking_20_manifest.jsonl \
  --predictions-path artifacts/mlx_whisper_large_v3_10h_500step_lora_last8_r8_merged_validation_banking_20_postprocessed_predictions.jsonl \
  --output-path artifacts/mlx_whisper_large_v3_10h_500step_lora_last8_r8_merged_validation_banking_20_postprocessed_eval.jsonl
```

Results:

| Model | Mode | Rows | WER | Entity error rate |
|---|---|---:|---:|---:|
| Baseline large-v3 | raw | 20 | 4.47% | 10.00% |
| Last-4 LoRA 10-step | raw | 20 | 4.47% | 10.00% |
| Last-8 LoRA 10-step | raw | 20 | 4.47% | 10.00% |
| Last-8 LoRA 500-step | raw | 20 | 4.47% | 10.00% |
| Last-8 LoRA 500-step | postprocessed | 20 | 1.12% | 0.00% |

## Observations

The wider last-8 adapter is stable and does not regress the 20-row banking subset, but it still does not alter the same smoke metrics versus baseline. The final loss alone is not enough evidence of ASR quality improvement.

## Decision

Before running full validation/test transcription, inspect whether last-8 500-step changes hypotheses beyond punctuation on the small subset. If small-subset outputs are identical to baseline, prioritize data/postprocessing or higher-rank LoRA instead of another same-shape scale-up.

Next action:

```text
compare last-8 500-step hypotheses against baseline and last-4 500-step on the same rows
```

