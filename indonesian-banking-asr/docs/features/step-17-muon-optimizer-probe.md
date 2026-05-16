---
title: "Step 17: Muon Optimizer Probe"
type: [feature-note, training, evaluation]
created: 2026-05-16
status: completed
categories: [asr, whisper, mlx, fine-tuning, optimizer, muon]
related:
  - step-16-mlx-large-v3-scaleup-experiments.md
  - step-15-mlx-large-v3-combined-finetune.md
  - ../../README.md
author: fauzan
---

# Step 17: Muon Optimizer Probe

Step ini mencoba Muon untuk mempercepat MLX Whisper large-v3 fine-tuning. Hasilnya: Muon tidak aman pada setup decoder-only saat ini.

## Implementation

Added optimizer choices to MLX trainer:

```text
--optimizer muon
--optimizer muon_sgd
```

`muon_sgd` uses `mlx.optimizers.MultiOptimizer`:

```text
Muon: 2D hidden weights only
SGD: token embedding, positional embedding, bias, LayerNorm, and other fallback params
```

## Raw Muon Probe

Command shape:

```bash
uv run banking-asr-finetune-mlx-whisper \
  --manifest-path data/training/combined_train_manifest_57.jsonl \
  --model-name mlx-community/whisper-large-v3-mlx \
  --max-steps 10 \
  --limit 10 \
  --train-scope decoder \
  --optimizer muon \
  --learning-rate 1e-8
```

Summary:

```json
{
  "optimizer": "muon",
  "learning_rate": 1e-8,
  "completed_steps": 10,
  "first_loss": 0.6587589383125305,
  "last_loss": NaN
}
```

Eval collapsed:

```text
BabelSpeech val WER: 100.00%
Synthetic banking test WER: 100.00%
Synthetic entity error rate: 100.00%
```

Generated text collapsed to repeated punctuation (`!`).

## Ultra-low LR Probe

Raw Muon at `1e-10` for 1 step kept finite loss but still collapsed generation on 1 real and 1 banking row:

```text
BabelSpeech 1-row WER: 100.00%
Synthetic banking 1-row WER: 100.00%
Synthetic banking entity error rate: 100.00%
```

## Hybrid Muon+SGD Probe

`muon_sgd` avoids applying Muon to embeddings, positional embeddings, bias, and LayerNorm. It still produced `NaN` loss by 10 steps at `1e-8`.

Summary:

```json
{
  "optimizer": "muon_sgd",
  "learning_rate": 1e-8,
  "completed_steps": 10,
  "first_loss": 0.6587589383125305,
  "last_loss": NaN
}
```

## Decision

Do not use Muon yet for this Whisper decoder fine-tune path.

Current safe optimizer remains:

```text
optimizer: sgd
learning_rate: 1e-8
train_scope: decoder
```

## Next Safer Speedups

Try speed without Muon first:

```text
sgd lr3e-8, 50 steps
sgd lr1e-7, 50 steps
```

If Muon is revisited, add gradient clipping and stricter parameter filtering before another run.
