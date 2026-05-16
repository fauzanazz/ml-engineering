---
title: "Step 19: Whisper-style AdamW Schedule Probe"
type: [feature-note, training, evaluation]
created: 2026-05-16
status: completed
categories: [asr, whisper, mlx, fine-tuning, adamw, schedule]
related:
  - step-18-guarded-optimizer-experiments.md
  - step-17-muon-optimizer-probe.md
  - ../../README.md
author: fauzan
external_references:
  - https://cdn.openai.com/papers/whisper.pdf
---

# Step 19: Whisper-style AdamW Schedule Probe

Step ini mencoba AdamW dengan schedule yang mengikuti pola training Whisper asli: AdamW, warmup, linear decay, weight decay, beta2 rendah, epsilon lebih besar, dan gradient clipping. Referensi: OpenAI Whisper paper, Appendix D.1 Training Details (`https://cdn.openai.com/papers/whisper.pdf`).

## Source Hyperparameter Pattern

Whisper paper reports these relevant training choices:

```text
optimizer: AdamW
beta2: 0.98
epsilon: 1e-6
weight decay: 0.1
gradient clipping: global norm 1.0
warmup updates: 2048
learning rate schedule: linear decay to zero
```

The original learning rates are pretraining-scale values, so this experiment keeps the optimizer/schedule shape but uses drastically smaller peak LR for tiny local fine-tuning.

## Implementation

Added to MLX trainer:

```text
--lr-schedule warmup_linear_decay
--weight-decay
--adam-beta1
--adam-beta2
--adam-eps
```

Schedule:

```text
warmup: lr = peak_lr * step / warmup_steps
decay: lr = peak_lr * (1 - progress_after_warmup)
```

Tests added for warmup + linear decay.

## Direct Paper-shape Probe

Command:

```bash
uv run banking-asr-finetune-mlx-whisper \
  --manifest-path data/training/combined_train_manifest_57.jsonl \
  --output-dir models/mlx-whisper-large-v3-combined-10step-adamw-whisper-paper-lr1e-12 \
  --summary-path artifacts/mlx_whisper_large_v3_combined_10step_adamw_whisper_paper_lr1e-12_summary.jsonl \
  --model-name mlx-community/whisper-large-v3-mlx \
  --max-steps 10 \
  --limit 10 \
  --train-scope decoder_last_4 \
  --optimizer adamw \
  --learning-rate 1e-12 \
  --warmup-steps 5 \
  --lr-schedule warmup_linear_decay \
  --max-grad-norm 1.0 \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  --adam-eps 1e-6
```

Summary:

```json
{
  "model_name": "mlx-community/whisper-large-v3-mlx",
  "rows_seen": 10,
  "max_steps": 10,
  "completed_steps": 10,
  "learning_rate": 1e-12,
  "warmup_steps": 5,
  "max_grad_norm": 1.0,
  "lr_schedule": "warmup_linear_decay",
  "weight_decay": 0.1,
  "adam_beta1": 0.9,
  "adam_beta2": 0.98,
  "adam_eps": 1e-6,
  "optimizer": "adamw",
  "train_scope": "decoder_last_4",
  "first_loss": 0.6587589383125305,
  "last_loss": 0.8432043790817261
}
```

Held-out smoke eval:

```text
BabelSpeech val 5-row WER: 4.55%
Synthetic banking test 5-row WER: 4.08%
Synthetic banking entity error rate: 0.00%
```

## Failed Higher-LR Probe

Same schedule shape with LR `1e-11`, default AdamW beta2/eps, and clipping `0.1` still produced `NaN`. Stable AdamW needed both:

```text
very low peak LR: 1e-12
larger epsilon: 1e-6
paper beta2: 0.98
linear decay to zero
```

## Interpretation

Whisper-style scheduling fixes the AdamW collapse only at ultra-low local fine-tune LR. That makes AdamW stable, but not useful yet: held-out WER is unchanged and last loss increased.

Likely cause: original Whisper settings assume huge batch/data scale and long training. This project has 57 mixed rows, batch size 1, and direct decoder updates. The optimizer shape transfers, but the absolute LR must be many orders smaller.

## Decision

Current best stable options:

```text
safe baseline: SGD lr1e-8 decoder, full decoder trainable
faster safe path: SGD lr3e-8 decoder_last_4 + warmup + clip
stable AdamW path: AdamW paper-style lr1e-12 decoder_last_4 + warmup + linear decay + clip
```

AdamW is now stable but too tiny to improve WER. Prefer guarded SGD for productive scale-up, or use adapter/LoRA instead of direct decoder updates for AdamW.
