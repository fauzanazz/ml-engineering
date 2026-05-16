---
title: "Step 24: MLX Whisper LoRA Smoke"
type: [feature-note, training, adapter]
created: 2026-05-16
status: completed
categories: [asr, whisper, mlx, lora, adapter, fine-tuning]
related:
  - step-23-entity-aware-postprocessing-10h.md
  - step-22-10h-baseline-evaluation.md
  - ../../README.md
author: fauzan
---

# Step 24: MLX Whisper LoRA Smoke

Step ini menambahkan parameter-efficient LoRA path untuk MLX Whisper large-v3. Tujuannya membuat training path yang lebih aman daripada direct decoder update, lalu menjalankan smoke test kecil di 10h training manifest.

## Implementation

Added `LoRALinear` wrapper in the MLX trainer and two train scopes:

```text
decoder_last_4_lora
decoder_last_8_lora
```

Scope behavior:

```text
freeze full model
wrap query/value projections in self-attention and cross-attention
train only LoRA A/B weights
merge LoRA delta into base Linear before checkpoint save
```

Merged save is required because `mlx_whisper.transcribe` expects the original Whisper module shape. Raw LoRA modules create incompatible checkpoint keys such as `base`, `lora_a`, and `lora_b`.

Files changed:

```text
src/indonesian_banking_asr/training/mlx_whisper_finetune.py
tests/training/test_mlx_whisper_finetune.py
```

## Smoke Training

Command:

```bash
uv run banking-asr-finetune-mlx-whisper \
  --manifest-path data/training/combined_train_manifest_10h_80_20_candidate.jsonl \
  --output-dir models/mlx-whisper-large-v3-10h-10step-lora-last4-r8-merged \
  --summary-path artifacts/mlx_whisper_large_v3_10h_10step_lora_last4_r8_merged_summary.jsonl \
  --model-name mlx-community/whisper-large-v3-mlx \
  --max-steps 10 \
  --limit 10 \
  --train-scope decoder_last_4_lora \
  --optimizer adamw \
  --learning-rate 1e-6 \
  --warmup-steps 2 \
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
  "rows_seen": 10,
  "completed_steps": 10,
  "learning_rate": 1e-06,
  "optimizer": "adamw",
  "train_scope": "decoder_last_4_lora",
  "lora_rank": 8,
  "lora_alpha": 16.0,
  "lora_modules": 16,
  "first_loss": 0.788602888584137,
  "last_loss": 0.5687903761863708
}
```

Training stayed stable and loss decreased.

## Loadability Check

The merged checkpoint was loaded by `banking-asr-transcribe-whisper` and evaluated on a 20-row banking validation smoke slice.

| Model | Rows | WER | Entity error rate |
|---|---:|---:|---:|
| baseline large-v3 | 20 | 4.47% | 10.00% |
| 10-step LoRA merged | 20 | 4.47% | 10.00% |

The smoke proves the LoRA path trains and saves loadable checkpoints. It does not show quality improvement yet, which is expected for 10 steps on 10 rows.

## Validation

Ran targeted tests:

```bash
uv run pytest tests/training/test_mlx_whisper_finetune.py
```

Result:

```text
5 passed
```

## Decision

LoRA is now the preferred training path for Step 25 scale-up. Direct decoder-only training remains a fallback, but LoRA gives a cleaner adapter-style path and avoids full decoder instability.

Recommended Step 25:

```text
scale decoder_last_4_lora on full 10h train manifest: 500-1000 steps, evaluate raw + postprocessed validation/test
```
