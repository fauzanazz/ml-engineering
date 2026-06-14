---
title: "Step 30: Banking-Only LoRA at lr=5e-5"
type: [feature-note, training, evaluation, hyperparameter]
created: 2026-05-17
status: completed
categories: [asr, whisper, mlx, lora, learning-rate]
related:
  - step-29-banking-only-lora-200step.md
  - step-28-last8-lora-500step-scaleup.md
author: fauzan
---

# Step 30: Banking-Only LoRA at lr=5e-5

Step 29 menunjukkan output transkrip tidak bergeser sama sekali pada lr=1e-6. Step ini menaikkan learning rate 50x menjadi 5e-5, dengan konfigurasi lain identik, untuk menguji apakah adapter benar-benar bisa memengaruhi inference.

## Training

Command:

```bash
uv run banking-asr-finetune-mlx-whisper \
  --manifest-path artifacts/synthetic_banking_train_manifest_10h_candidate.jsonl \
  --output-dir models/mlx-whisper-large-v3-banking-only-200step-lora-last8-r8-lr5e-5-merged \
  --summary-path artifacts/mlx_whisper_large_v3_banking_only_200step_lora_last8_r8_lr5e-5_merged_summary.jsonl \
  --model-name mlx-community/whisper-large-v3-mlx \
  --max-steps 200 \
  --train-scope decoder_last_8_lora \
  --optimizer adamw \
  --learning-rate 5e-5 \
  --warmup-steps 20 \
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
  "rows_seen": 1008,
  "completed_steps": 200,
  "learning_rate": 5e-05,
  "first_loss": 0.7034010291099548,
  "last_loss": 0.1520625799894333
}
```

Loss drops sharply from 0.70 to 0.15 in 200 steps, much faster than lr=1e-6.

## Evaluation

Validation banking 20-row subset:

| Mode | Rows | WER | Entity error rate |
|---|---:|---:|---:|
| Baseline raw | 20 | 4.47% | 10.00% |
| LoRA lr=5e-5 raw | 20 | 3.91% | 8.33% |
| LoRA lr=5e-5 postprocessed | 20 | 1.12% | 0.00% |

Raw WER and entity error rate moved for the first time across Steps 24–29. The diff against baseline shows one CARD_LAST4 fix:

```text
syn_id_credit_card_limit_001_000112_s03_p01
 base:  ... untuk kartu akhir 9.333?
 lora:  ... untuk kartu akhir 9333?
```

## Decision

lr=5e-5 is the correct order of magnitude for LoRA on this stack: training is stable, loss decreases, and raw outputs start to absorb banking numeric formatting. Next step pushes lr=1e-4 to see if the gains are linear or if 5e-5 already sits near the productive ceiling for the same scope and dataset.

Next action:

```text
banking-only LoRA, decoder_last_8, lr=1e-4 for 200 steps, evaluate same 20-row banking subset, then expand to full 10h splits if quality improves
```

