---
title: "Step 27: Last-8 LoRA Smoke"
type: [feature-note, training, evaluation]
created: 2026-05-17
status: completed
categories: [asr, whisper, mlx, lora, smoke-test]
related:
  - step-26-baseline-vs-lora-10h-slice-comparison.md
  - step-25-10h-lora-scaleup-evaluation.md
author: fauzan
---

# Step 27: Last-8 LoRA Smoke

Step ini mencoba variabel lebih kuat dari Step 26: LoRA pada 8 decoder block terakhir, bukan 4 decoder block terakhir. Tujuannya memastikan konfigurasi lebih lebar stabil dan loadable sebelum scale-up.

## Training

Command:

```bash
uv run banking-asr-finetune-mlx-whisper \
  --manifest-path data/training/combined_train_manifest_10h_80_20_candidate.jsonl \
  --output-dir models/mlx-whisper-large-v3-10h-10step-lora-last8-r8-merged \
  --summary-path artifacts/mlx_whisper_large_v3_10h_10step_lora_last8_r8_merged_summary.jsonl \
  --model-name mlx-community/whisper-large-v3-mlx \
  --max-steps 10 \
  --train-scope decoder_last_8_lora \
  --optimizer adamw \
  --learning-rate 1e-6 \
  --warmup-steps 2 \
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
  "completed_steps": 10,
  "learning_rate": 1e-06,
  "warmup_steps": 2,
  "lr_schedule": "warmup_linear_decay",
  "optimizer": "adamw",
  "train_scope": "decoder_last_8_lora",
  "lora_rank": 8,
  "lora_alpha": 16.0,
  "lora_modules": 32,
  "first_loss": 0.788602888584137,
  "last_loss": 0.5688133835792542
}
```

Loss decreased over 10 steps. Checkpoint saved and loaded successfully through `banking-asr-transcribe-whisper`.

## Evaluation

Validation subset:

```text
artifacts/combined_10h_validation_banking_20_manifest.jsonl
```

Commands:

```bash
uv run banking-asr-transcribe-whisper \
  --manifest-path artifacts/combined_10h_validation_banking_20_manifest.jsonl \
  --model models/mlx-whisper-large-v3-10h-10step-lora-last8-r8-merged \
  --output-path artifacts/mlx_whisper_large_v3_10h_10step_lora_last8_r8_merged_validation_banking_20_predictions.jsonl \
  --language id

uv run banking-asr-evaluate \
  --manifest-path artifacts/combined_10h_validation_banking_20_manifest.jsonl \
  --predictions-path artifacts/mlx_whisper_large_v3_10h_10step_lora_last8_r8_merged_validation_banking_20_predictions.jsonl \
  --output-path artifacts/mlx_whisper_large_v3_10h_10step_lora_last8_r8_merged_validation_banking_20_eval.jsonl

uv run banking-asr-postprocess-predictions \
  --predictions-path artifacts/mlx_whisper_large_v3_10h_10step_lora_last8_r8_merged_validation_banking_20_predictions.jsonl \
  --output-path artifacts/mlx_whisper_large_v3_10h_10step_lora_last8_r8_merged_validation_banking_20_postprocessed_predictions.jsonl

uv run banking-asr-evaluate \
  --manifest-path artifacts/combined_10h_validation_banking_20_manifest.jsonl \
  --predictions-path artifacts/mlx_whisper_large_v3_10h_10step_lora_last8_r8_merged_validation_banking_20_postprocessed_predictions.jsonl \
  --output-path artifacts/mlx_whisper_large_v3_10h_10step_lora_last8_r8_merged_validation_banking_20_postprocessed_eval.jsonl
```

Results:

| Model | Mode | Rows | WER | Entity error rate |
|---|---|---:|---:|---:|
| Baseline large-v3 | raw | 20 | 4.47% | 10.00% |
| Last-4 LoRA 10-step | raw | 20 | 4.47% | 10.00% |
| Last-8 LoRA 10-step | raw | 20 | 4.47% | 10.00% |
| Last-8 LoRA 10-step | postprocessed | 20 | 1.12% | 0.00% |

## Decision

Last-8 LoRA is technically safe: training, merge-save, loading, transcription, and evaluation all work. The 10-step smoke does not yet change output quality versus baseline or last-4 smoke.

Next experiment should scale this exact stable configuration to 500 steps and compare against Step 25:

```text
decoder_last_8_lora, rank 8, alpha 16, 500 steps, same 10h validation/test slices
```

