---
title: "Step 29: Banking-Only LoRA 200-Step"
type: [feature-note, training, evaluation, data]
created: 2026-05-17
status: completed
categories: [asr, whisper, mlx, lora, data-mix]
related:
  - step-28-last8-lora-500step-scaleup.md
  - step-27-last8-lora-smoke.md
  - step-25-10h-lora-scaleup-evaluation.md
author: fauzan
---

# Step 29: Banking-Only LoRA 200-Step

Step ini menguji hipotesis bahwa sinyal banking di manifest 10h dilemahkan oleh 2,831 baris BabelSpeech non-banking. Eksperimen melatih LoRA hanya pada 1,008 baris synthetic banking, dengan harapan adapter mau benar-benar bergerak pada entity banking.

## Manifest

Filtered training manifest:

```bash
python - <<'PY'
import json
rows = [json.loads(l) for l in open('data/training/combined_train_manifest_10h_80_20_candidate.jsonl')]
banking = [r for r in rows if r.get('source') == 'template_gemini']
with open('artifacts/synthetic_banking_train_manifest_10h_candidate.jsonl', 'w') as out:
    for r in banking:
        out.write(json.dumps(r, ensure_ascii=False) + '\n')
PY
```

Result:

```text
artifacts/synthetic_banking_train_manifest_10h_candidate.jsonl  rows=1008
```

## Training

Command:

```bash
uv run banking-asr-finetune-mlx-whisper \
  --manifest-path artifacts/synthetic_banking_train_manifest_10h_candidate.jsonl \
  --output-dir models/mlx-whisper-large-v3-banking-only-200step-lora-last8-r8-merged \
  --summary-path artifacts/mlx_whisper_large_v3_banking_only_200step_lora_last8_r8_merged_summary.jsonl \
  --model-name mlx-community/whisper-large-v3-mlx \
  --max-steps 200 \
  --train-scope decoder_last_8_lora \
  --optimizer adamw \
  --learning-rate 1e-6 \
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
  "learning_rate": 1e-06,
  "warmup_steps": 20,
  "lr_schedule": "warmup_linear_decay",
  "optimizer": "adamw",
  "train_scope": "decoder_last_8_lora",
  "lora_rank": 8,
  "lora_alpha": 16.0,
  "lora_modules": 32,
  "first_loss": 0.7034010291099548,
  "last_loss": 0.582604706287384
}
```

Loss declines but stays much higher than the 10h LoRA runs because banking-only rows are denser in numeric entities.

## Evaluation

Same 20-row banking subset:

| Mode | Rows | WER | Entity error rate |
|---|---:|---:|---:|
| Raw | 20 | 4.47% | 10.00% |
| Postprocessed | 20 | 1.12% | 0.00% |

Hypotheses are byte-identical to baseline on the smoke set.

## Observations and Decision

Across last-4 LoRA 10/500-step, last-8 LoRA 10/500-step, and banking-only last-8 LoRA 200-step, raw smoke outputs do not move at all from baseline. The shared knob is `--learning-rate 1e-6` with very small warmup. Combined with frozen base weights and tiny LoRA initialization, the adapter contribution is effectively a no-op at inference temperature 0.

Next experiment must change the learning rate regime, not the scope or data mix:

```text
banking-only LoRA, decoder_last_8, learning_rate sweep at 5e-5 and 1e-4 for 200 steps, evaluate same smoke subset
```

