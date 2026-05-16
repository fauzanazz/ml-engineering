---
title: "Step 14: Whisper Tiny Combined Fine-tune Smoke"
type: [feature-note, training, evaluation]
created: 2026-05-16
status: completed
categories: [asr, whisper, fine-tuning, training, babelspeech, synthetic-banking]
related:
  - step-13-combined-training-manifest.md
  - step-12-babelspeech-real-data-grounding.md
  - ../../README.md
author: fauzan
---

# Step 14: Whisper Tiny Combined Fine-tune Smoke

Step ini menjalankan fine-tune smoke pada combined training manifest. Ini membuktikan training loop bisa berjalan end-to-end di Apple Silicon MPS, bukan final production fine-tune.

## Implementation

Added:

```text
src/indonesian_banking_asr/training/whisper_finetune.py
tests/training/test_whisper_finetune.py
```

CLI:

```text
banking-asr-finetune-whisper
```

Dependencies added:

```text
transformers
accelerate
soundfile
jiwer
```

## Training Run

Command:

```bash
uv run banking-asr-finetune-whisper \
  --manifest-path data/training/combined_train_manifest_57.jsonl \
  --output-dir models/whisper-tiny-combined-10step \
  --summary-path artifacts/whisper_tiny_combined_10step_summary.jsonl \
  --model-name openai/whisper-tiny \
  --max-steps 10 \
  --batch-size 1 \
  --limit 20
```

Summary:

```json
{
  "model_name": "openai/whisper-tiny",
  "output_dir": "models/whisper-tiny-combined-10step",
  "rows_seen": 20,
  "max_steps": 10,
  "completed_steps": 10,
  "batch_size": 1,
  "learning_rate": 0.00001,
  "device": "mps",
  "first_loss": 3.518184185028076,
  "last_loss": 3.9010555744171143
}
```

Generated checkpoint:

```text
models/whisper-tiny-combined-10step
```

Generated artifacts:

```text
artifacts/whisper_tiny_combined_10step_summary.jsonl
artifacts/whisper_tiny_combined_10step_babelspeech_val_manifest.jsonl
artifacts/whisper_tiny_combined_10step_babelspeech_val_predictions.jsonl
artifacts/whisper_tiny_combined_10step_babelspeech_val_eval.jsonl
```

## Validation Eval

Eval target: 5 BabelSpeech validation rows from the 50-row sample.

Result:

```json
{
  "rows": 5,
  "word_errors": 46,
  "reference_words": 88,
  "wer": 0.5227272727272727,
  "entity_errors": 0,
  "entities": 0,
  "entity_error_rate": 0.0
}
```

Interpretation:

- This is **52.27% WER** on tiny-model 5-row validation smoke.
- It is not comparable to prior MLX Whisper large-v3 baseline because model family changed from `mlx-community/whisper-large-v3-mlx` to `openai/whisper-tiny`.
- Tiny smoke fine-tune proves pipeline execution; it does not prove quality improvement.
- Last loss increased vs first loss, so 10 steps on 20 rows is not a stable training run.

## Existing Strong Baselines Remain

```text
MLX Whisper large-v3 synthetic clean+aug raw WER: 15.76%
MLX Whisper large-v3 synthetic clean+aug post-processed WER: 10.30%
MLX Whisper large-v3 BabelSpeech sample raw WER: 11.93%
Whisper tiny combined smoke BabelSpeech val WER: 52.27%
```

## Next Step

For meaningful combined training, run a full fine-tune with a comparable model size or LoRA path, then evaluate:

- BabelSpeech held-out validation/test for general Indonesian regression.
- Synthetic banking held-out test for WER and entity error rate.
- Raw and post-processed outputs separately.
