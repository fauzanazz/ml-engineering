---
title: "Step 16: MLX Whisper Large-v3 Scale-up Experiments"
type: [feature-note, training, evaluation]
created: 2026-05-16
status: completed
categories: [asr, whisper, mlx, fine-tuning, experiment]
related:
  - step-15-mlx-large-v3-combined-finetune.md
  - step-13-combined-training-manifest.md
  - ../../README.md
author: fauzan
---

# Step 16: MLX Whisper Large-v3 Scale-up Experiments

Step ini melanjutkan MLX-native fine-tune dari Step 15 dengan scale-up bertahap: 10, 50, dan 200 decoder-only SGD steps pada combined training manifest.

## Setup

Shared config:

```text
model: mlx-community/whisper-large-v3-mlx
training manifest: data/training/combined_train_manifest_57.jsonl
train scope: decoder
optimizer: sgd
learning rate: 1e-8
```

Held-out eval slices:

```text
BabelSpeech validation: 5 rows, 88 reference words
Synthetic banking test: 5 rows, 49 reference words, 15 entities
```

## Results

| Experiment | Rows seen | Steps | First loss | Last loss | BabelSpeech val WER | Synthetic test WER | Synthetic entity error rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| save-only baseline | - | 0 | - | - | 4.55% | 4.08% | 0.00% |
| 1-step SGD lr1e-8 | 1 | 1 | 0.6588 | 0.6588 | 4.55% | - | - |
| 10-step SGD lr1e-8 | 10 | 10 | 0.6588 | 0.8432 | 4.55% | 4.08% | 0.00% |
| 50-step SGD lr1e-8 | 57 | 50 | 0.6588 | 0.3127 | 4.55% | 4.08% | 0.00% |
| 200-step SGD lr1e-8 | 57 | 200 | 0.6588 | 0.5968 | 4.55% | 4.08% | 0.00% |

## Artifacts

Experiment summary:

```text
artifacts/mlx_whisper_large_v3_combined_experiment_summary.jsonl
```

Checkpoints:

```text
models/mlx-whisper-large-v3-combined-10step-sgd-lr1e-8
models/mlx-whisper-large-v3-combined-50step-sgd-lr1e-8
models/mlx-whisper-large-v3-combined-200step-sgd-lr1e-8
```

Eval artifacts:

```text
artifacts/mlx_whisper_large_v3_combined_10step_sgd_lr1e-8_babelspeech_val_eval.jsonl
artifacts/mlx_whisper_large_v3_combined_10step_sgd_lr1e-8_synthetic_test_5_eval.jsonl
artifacts/mlx_whisper_large_v3_combined_50step_sgd_lr1e-8_babelspeech_val_eval.jsonl
artifacts/mlx_whisper_large_v3_combined_50step_sgd_lr1e-8_synthetic_test_5_eval.jsonl
artifacts/mlx_whisper_large_v3_combined_200step_sgd_lr1e-8_babelspeech_val_eval.jsonl
artifacts/mlx_whisper_large_v3_combined_200step_sgd_lr1e-8_synthetic_test_5_eval.jsonl
```

## Interpretation

The conservative SGD path is stable: no repeated punctuation collapse and no held-out degradation on these small eval slices. However, it also shows no measurable WER improvement versus the save-only baseline.

Likely reasons:

- Eval slices are tiny and already easy for large-v3.
- Learning rate `1e-8` is intentionally conservative.
- Decoder-only training limits adaptation.
- Combined manifest has only 57 train rows.

## Next Experiment

Use the same stable decoder-only path, but increase learning rate carefully:

```text
sgd lr3e-8, 50 steps
sgd lr1e-7, 50 steps
```

Stop if repeated punctuation or BabelSpeech WER regression appears. If stable, expand eval slices from 5 rows to full held-out validation/test rows.
