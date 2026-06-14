---
title: "Step 37: Full-Mix Rank-4 LoRA Regularization"
type: [feature-note, training, evaluation, regression-analysis]
created: 2026-05-17
status: completed
categories: [asr, whisper, mlx, lora, rank-ablation]
related:
  - step-36-fullmix-last4-lora-scope-probe.md
  - step-33-fullmix-lora-lr1e-4.md
author: fauzan
---

# Step 37: Full-Mix Rank-4 LoRA Regularization

Step 36 menunjukkan scope last-4 memperburuk WER. Step ini kembali ke konfigurasi
terbaik Step 33 (`decoder_last_8_lora`, full-mix, 200 step, lr=1e-4), tetapi
menurunkan kapasitas LoRA dari rank 8 alpha 16 ke rank 4 alpha 8 agar rasio
alpha/rank tetap 2. Hipotesis: low-rank regularization mengurangi regresi real
tanpa menghapus gain banking.

## Training

```bash
uv run banking-asr-finetune-mlx-whisper \
  --manifest-path data/training/combined_train_manifest_10h_80_20_candidate.jsonl \
  --output-dir models/mlx-whisper-large-v3-fullmix-200step-lora-last8-r4-a8-lr1e-4-merged \
  --summary-path artifacts/mlx_whisper_large_v3_fullmix_200step_lora_last8_r4_a8_lr1e-4_merged_summary.jsonl \
  --model-name mlx-community/whisper-large-v3-mlx \
  --max-steps 200 \
  --train-scope decoder_last_8_lora \
  --optimizer adamw \
  --learning-rate 1e-4 \
  --lora-rank 4 \
  --lora-alpha 8
```

Summary: `rows_seen=3839`, `completed_steps=200`, `first_loss=0.789`,
`last_loss=0.0090`, `lora_rank=4`, `lora_alpha=8`, `lora_modules=32`.

## Evaluation

```bash
P=artifacts/mlx_whisper_large_v3_fullmix_200step_lora_last8_r4_a8_lr1e-4_merged
M=models/mlx-whisper-large-v3-fullmix-200step-lora-last8-r4-a8-lr1e-4-merged
uv run banking-asr-transcribe-whisper --manifest-path artifacts/combined_10h_validation_manifest.jsonl --model $M --output-path ${P}_validation_predictions.jsonl --language id
uv run banking-asr-transcribe-whisper --manifest-path artifacts/combined_10h_test_manifest.jsonl --model $M --output-path ${P}_test_predictions.jsonl --language id
for split in validation test; do
  uv run banking-asr-postprocess-predictions --predictions-path ${P}_${split}_predictions.jsonl --output-path ${P}_${split}_postprocessed_predictions.jsonl
  for slice in "" "_banking" "_real"; do
    man=artifacts/combined_10h_${split}${slice}_manifest.jsonl
    uv run banking-asr-evaluate --manifest-path $man --predictions-path ${P}_${split}_predictions.jsonl --output-path ${P}_${split}${slice}_eval.jsonl
    uv run banking-asr-evaluate --manifest-path $man --predictions-path ${P}_${split}_postprocessed_predictions.jsonl --output-path ${P}_${split}${slice}_postprocessed_eval.jsonl
  done
done
```

## Results

Raw WER / entity-error (%). Columns: baseline / rank-8 Step 33 / rank-4 this.

| Split | Slice | Base WER | S33 WER | S37 WER | Base Ent | S33 Ent | S37 Ent |
|---|---|---|---|---|---|---|---|
| val | all | 12.11 | 13.19 | 13.19 | 17.27 | 6.47 | 7.91 |
| val | banking | 9.69 | 5.58 | 5.91 | 17.27 | 6.47 | 7.91 |
| val | real | 12.51 | 14.45 | 14.39 | 0.00 | 0.00 | 0.00 |
| test | all | 13.02 | 12.75 | 12.33 | 13.96 | 7.08 | 7.92 |
| test | banking | 14.23 | 5.87 | 6.08 | 13.96 | 7.08 | 7.92 |
| test | real | 12.80 | 13.98 | 13.46 | 0.00 | 0.00 | 0.00 |

Postprocessed banking metrics (%):

| Split | Base WER | S33 WER | S37 WER | Base Ent | S33 Ent | S37 Ent |
|---|---|---|---|---|---|---|
| val banking | 4.19 | 5.09 | 4.93 | 1.44 | 5.04 | 5.04 |
| test banking | 10.30 | 5.01 | 5.08 | 3.33 | 4.58 | 5.00 |

Observasi:
- Rank-4 adalah regularizer terbaik sejauh ini: test real membaik dari Step 33 (13.98 -> 13.46) dan validation real sedikit membaik (14.45 -> 14.39).
- Banking tetap kuat walau sedikit turun: raw WER val 5.58 -> 5.91, test 5.87 -> 6.08, masih jauh lebih baik dari baseline.
- Test all raw/postprocessed lebih baik dari baseline (12.33/12.18 vs 13.02/12.43), karena gain banking melebihi real regression.
- Entity raw memburuk dibanding rank-8 (6.47/7.08 -> 7.91/7.92), tapi masih lebih baik dari baseline raw.

## Decision

```
Rank-4 low-rank regularization DITERIMA sebagai trade-off lebih baik daripada
rank-8 untuk retensi real, terutama pada test. Namun real masih belum kembali ke
baseline (val 14.39 vs 12.51, test 13.46 vs 12.80).

Next (Step 38): coba rank lebih rendah (`--lora-rank 2`, `--lora-alpha 4`) pada
full-mix 200-step last-8 lr=1e-4. Target: real regression turun lagi dengan
banking raw WER tetap <7.0.
```
