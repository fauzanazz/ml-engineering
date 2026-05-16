---
title: "Step 18: Guarded Optimizer Experiments"
type: [feature-note, training, evaluation]
created: 2026-05-16
status: completed
categories: [asr, whisper, mlx, fine-tuning, optimizer, stability]
related:
  - step-17-muon-optimizer-probe.md
  - step-16-mlx-large-v3-scaleup-experiments.md
  - ../../README.md
author: fauzan
---

# Step 18: Guarded Optimizer Experiments

Step ini menambahkan safety controls ke MLX Whisper trainer: partial unfreeze, linear warmup, dan gradient clipping. Tujuannya menguji apakah AdamW/Muon bisa dibuat stabil, dan apakah SGD bisa dipercepat tanpa collapse.

## Implementation

Trainer updates:

```text
src/indonesian_banking_asr/training/mlx_whisper_finetune.py
```

New flags:

```text
--train-scope decoder_last_4
--train-scope decoder_last_8
--warmup-steps N
--max-grad-norm X
```

Freeze behavior:

```text
decoder: all decoder params trainable
decoder_last_4: encoder frozen, decoder embeddings frozen, only last 4 decoder blocks + final decoder LN trainable
decoder_last_8: encoder frozen, decoder embeddings frozen, only last 8 decoder blocks + final decoder LN trainable
full: all params trainable
```

Gradient safety:

```text
mlx.optimizers.clip_grad_norm(grads, max_grad_norm)
```

Warmup:

```text
lr(step) = base_lr * min(step / warmup_steps, 1.0)
```

## Guarded Muon Result

Raw Muon with safeguards:

```bash
uv run banking-asr-finetune-mlx-whisper \
  --manifest-path data/training/combined_train_manifest_57.jsonl \
  --output-dir models/mlx-whisper-large-v3-combined-10step-muon-last4-warmup-clip \
  --summary-path artifacts/mlx_whisper_large_v3_combined_10step_muon_last4_warmup_clip_summary.jsonl \
  --model-name mlx-community/whisper-large-v3-mlx \
  --max-steps 10 \
  --limit 10 \
  --train-scope decoder_last_4 \
  --optimizer muon \
  --learning-rate 1e-8 \
  --warmup-steps 5 \
  --max-grad-norm 0.1
```

Result:

```json
{
  "optimizer": "muon",
  "train_scope": "decoder_last_4",
  "learning_rate": 1e-8,
  "warmup_steps": 5,
  "max_grad_norm": 0.1,
  "first_loss": 0.6587589383125305,
  "last_loss": NaN
}
```

Conclusion: partial unfreeze + warmup + clipping did not stabilize Muon.

## Guarded AdamW Result

AdamW with stronger LR reduction:

```bash
uv run banking-asr-finetune-mlx-whisper \
  --manifest-path data/training/combined_train_manifest_57.jsonl \
  --output-dir models/mlx-whisper-large-v3-combined-10step-adamw-last4-lr1e-10-warmup-clip \
  --summary-path artifacts/mlx_whisper_large_v3_combined_10step_adamw_last4_lr1e-10_warmup_clip_summary.jsonl \
  --model-name mlx-community/whisper-large-v3-mlx \
  --max-steps 10 \
  --limit 10 \
  --train-scope decoder_last_4 \
  --optimizer adamw \
  --learning-rate 1e-10 \
  --warmup-steps 5 \
  --max-grad-norm 0.1
```

Result:

```json
{
  "optimizer": "adamw",
  "train_scope": "decoder_last_4",
  "learning_rate": 1e-10,
  "warmup_steps": 5,
  "max_grad_norm": 0.1,
  "first_loss": 0.6587589383125305,
  "last_loss": NaN
}
```

Conclusion: AdamW also remains unstable for this direct decoder fine-tune setup.

## Hybrid Muon+SGD Result

Hybrid `muon_sgd` attempted to use Muon for 2D hidden weights and SGD for embeddings/bias/LayerNorm fallback. MLX `MultiOptimizer` hit a tree merge error on this nested Whisper parameter tree:

```text
ValueError: Trees contain elements at the same locations but no merge function was provided
```

Conclusion: `muon_sgd` needs custom parameter-tree splitting before it is usable here.

## Faster Guarded SGD Result

SGD with higher LR plus safeguards stayed stable:

```bash
uv run banking-asr-finetune-mlx-whisper \
  --manifest-path data/training/combined_train_manifest_57.jsonl \
  --output-dir models/mlx-whisper-large-v3-combined-50step-sgd-last4-lr3e-8-warmup-clip \
  --summary-path artifacts/mlx_whisper_large_v3_combined_50step_sgd_last4_lr3e-8_warmup_clip_summary.jsonl \
  --model-name mlx-community/whisper-large-v3-mlx \
  --max-steps 50 \
  --limit 57 \
  --train-scope decoder_last_4 \
  --optimizer sgd \
  --learning-rate 3e-8 \
  --warmup-steps 10 \
  --max-grad-norm 0.1
```

Training summary:

```json
{
  "optimizer": "sgd",
  "train_scope": "decoder_last_4",
  "learning_rate": 3e-8,
  "warmup_steps": 10,
  "max_grad_norm": 0.1,
  "first_loss": 0.6587589383125305,
  "last_loss": 0.3126668930053711
}
```

Held-out smoke eval:

```text
BabelSpeech val 5-row WER: 4.55%
Synthetic banking test 5-row WER: 4.08%
Synthetic banking entity error rate: 0.00%
```

## Findings

- Muon remains unstable even after freezing embeddings, warmup, and clipping.
- AdamW remains unstable even at `1e-10` after freezing embeddings, warmup, and clipping.
- Faster SGD (`3e-8`) is stable under `decoder_last_4` + warmup + clipping.
- No WER gain yet on tiny held-out slices, but no regression either.

## Next Step

Use guarded SGD, not Muon/AdamW, for next scale-up:

```text
train_scope: decoder_last_4
optimizer: sgd
learning_rate: 3e-8
warmup_steps: 10
max_grad_norm: 0.1
max_steps: 200
```

Then evaluate larger held-out slices before trying higher LR.
