---
title: "Step 15: MLX Whisper Large-v3 Combined Fine-tune Smoke"
type: [feature-note, training, evaluation]
created: 2026-05-16
status: completed
categories: [asr, whisper, mlx, fine-tuning, training, babelspeech, synthetic-banking]
related:
  - step-14-whisper-tiny-combined-finetune.md
  - step-13-combined-training-manifest.md
  - ../../README.md
author: fauzan
---

# Step 15: MLX Whisper Large-v3 Combined Fine-tune Smoke

Step ini corrects Step 14 direction: MLX can train through `mlx.core`, `mlx.nn`, and `mlx.optimizers`. `mlx-whisper` does not ship a high-level trainer, but its model is an `mlx.nn.Module`, so a project-owned training loop can fine-tune it.

## Implementation

Added MLX-native Whisper fine-tune runner:

```text
src/indonesian_banking_asr/training/mlx_whisper_finetune.py
tests/training/test_mlx_whisper_finetune.py
```

CLI:

```text
banking-asr-finetune-mlx-whisper
```

Training loop:

```text
load mlx-whisper model
freeze encoder, train decoder
build log-mel features
build decoder teacher-forcing tokens
compute cross entropy with mlx.nn.losses.cross_entropy
compute gradients with mlx.nn.value_and_grad
update with mlx.optimizers
save MLX checkpoint with config + weights.safetensors
```

## Stable Large-v3 Smoke

Command:

```bash
uv run banking-asr-finetune-mlx-whisper \
  --manifest-path data/training/combined_train_manifest_57.jsonl \
  --output-dir models/mlx-whisper-large-v3-combined-1step-sgd-lr1e-8 \
  --summary-path artifacts/mlx_whisper_large_v3_combined_1step_sgd_lr1e-8_summary.jsonl \
  --model-name mlx-community/whisper-large-v3-mlx \
  --max-steps 1 \
  --limit 1 \
  --train-scope decoder \
  --optimizer sgd \
  --learning-rate 1e-8
```

Summary:

```json
{
  "model_name": "mlx-community/whisper-large-v3-mlx",
  "output_dir": "models/mlx-whisper-large-v3-combined-1step-sgd-lr1e-8",
  "rows_seen": 1,
  "max_steps": 1,
  "completed_steps": 1,
  "learning_rate": 1e-8,
  "optimizer": "sgd",
  "train_scope": "decoder",
  "first_loss": 0.6587589383125305,
  "last_loss": 0.6587589383125305
}
```

Validation smoke on 5 BabelSpeech validation rows:

```json
{
  "rows": 5,
  "word_errors": 4,
  "reference_words": 88,
  "wer": 0.045454545454545456,
  "entity_errors": 0,
  "entities": 0,
  "entity_error_rate": 0.0
}
```

Checkpoint:

```text
models/mlx-whisper-large-v3-combined-1step-sgd-lr1e-8
```

Artifacts:

```text
artifacts/mlx_whisper_large_v3_combined_1step_sgd_lr1e-8_summary.jsonl
artifacts/mlx_whisper_large_v3_combined_1step_sgd_lr1e-8_babelspeech_val_predictions.jsonl
artifacts/mlx_whisper_large_v3_combined_1step_sgd_lr1e-8_babelspeech_val_eval.jsonl
```

## Failed Optimizer Probe

AdamW with one decoder step destabilized generation, even at low learning rate, producing repeated punctuation and 100% WER on the same 5-row validation smoke. Save/load without updates was healthy, so the issue is optimizer/update scale, not checkpoint serialization.

Use conservative SGD for the next MLX fine-tune probes.

## Current Interpretation

This is still a smoke run, not a real trained model:

- Only 1 row seen.
- Only 1 update step.
- Eval set is 5 BabelSpeech validation rows.
- Decoder-only update chosen for safety.

But it confirms the correct target path: **MLX Whisper large-v3 can be fine-tuned with a project-owned MLX loop**.

## Next Step

Scale cautiously:

```text
optimizer: sgd
learning_rate: 1e-8 or lower
train_scope: decoder
max_steps: 10 -> 50 -> 200
```

After each scale-up, evaluate:

- BabelSpeech validation/test WER.
- Synthetic banking test raw WER.
- Synthetic banking test post-processed WER and entity error rate.
