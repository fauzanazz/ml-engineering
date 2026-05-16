---
title: "Step 20: Guarded SGD Scale-up Evaluation"
type: [feature-note, training, evaluation]
created: 2026-05-16
status: completed
categories: [asr, whisper, mlx, fine-tuning, sgd, evaluation]
related:
  - step-19-whisper-style-adamw-schedule.md
  - step-18-guarded-optimizer-experiments.md
  - ../../README.md
author: fauzan
---

# Step 20: Guarded SGD Scale-up Evaluation

Step ini scale guarded SGD path yang paling stabil dari Step 18-19. Tujuannya mengecek apakah `decoder_last_4` dengan LR kecil mulai mengubah held-out WER saat training diperpanjang dari 50 ke 200 steps.

## Training Setup

Command:

```bash
uv run banking-asr-finetune-mlx-whisper \
  --manifest-path data/training/combined_train_manifest_57.jsonl \
  --output-dir models/mlx-whisper-large-v3-combined-200step-sgd-last4-lr3e-8-warmup-clip \
  --summary-path artifacts/mlx_whisper_large_v3_combined_200step_sgd_last4_lr3e-8_warmup_clip_summary.jsonl \
  --model-name mlx-community/whisper-large-v3-mlx \
  --max-steps 200 \
  --train-scope decoder_last_4 \
  --optimizer sgd \
  --learning-rate 3e-8 \
  --warmup-steps 20 \
  --max-grad-norm 1.0
```

Summary:

```json
{
  "rows_seen": 57,
  "max_steps": 200,
  "completed_steps": 200,
  "learning_rate": 3e-08,
  "warmup_steps": 20,
  "max_grad_norm": 1.0,
  "optimizer": "sgd",
  "train_scope": "decoder_last_4",
  "first_loss": 0.6587589383125305,
  "last_loss": 0.5967848896980286
}
```

Loss decreased, so training stayed numerically stable.

## Evaluation Manifests

Created non-train eval slices:

```text
artifacts/babelspeech_eval_nontrain_manifest.jsonl: 10 / 50 rows
artifacts/synthetic_banking_eval_nontrain_manifest.jsonl: 108 / 450 rows
```

Rows were selected with `split != train` from the available BabelSpeech sample and synthetic banking dataset.

## Results

Baseline MLX Whisper large-v3 and 200-step fine-tuned checkpoint produced identical aggregate metrics on both eval slices.

| Model | Eval set | Rows | WER | Entity error rate | Entity errors |
|---|---:|---:|---:|---:|---:|
| Baseline large-v3 | BabelSpeech non-train | 10 | 15.49% | 0.00% | 0 / 0 |
| 200-step SGD last4 | BabelSpeech non-train | 10 | 15.49% | 0.00% | 0 / 0 |
| Baseline large-v3 | Synthetic banking non-train | 108 | 12.93% | 10.21% | 34 / 333 |
| 200-step SGD last4 | Synthetic banking non-train | 108 | 12.93% | 10.21% | 34 / 333 |

Synthetic banking entity errors by type:

```json
{"ACCOUNT_NUMBER": 12, "AMOUNT": 13, "PRODUCT_NAME": 9}
```

## Interpretation

Guarded SGD remains stable, but `decoder_last_4` updates at `3e-8` do not produce measurable ASR behavior changes after 200 steps. This confirms that the local direct-decoder path is too conservative for useful adaptation, despite stable loss movement.

The identical metrics suggest one of two likely causes:

- Updates are too small to affect decoding outputs.
- Last-4 decoder updates are not enough capacity for domain/entity correction.

## Decision

Stop spending more cycles on direct decoder-only optimizer probing unless adding better diagnostics. Next productive path is adapter-style fine-tuning or decoding-time/entity post-processing improvements.

Recommended Step 21:

```text
adapter/LoRA-style parameter-efficient training, or
entity-aware decoding/post-processing report on full synthetic eval
```
