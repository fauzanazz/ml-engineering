---
title: "Step 38: Full-Mix Rank-2 LoRA Regularization (Mixed Result)"
type: [feature-note, training, evaluation, regression-analysis]
created: 2026-05-17
status: completed
categories: [asr, whisper, mlx, lora, rank-ablation]
related:
  - step-37-fullmix-rank4-lora-regularization.md
  - step-33-fullmix-lora-lr1e-4.md
author: fauzan
---

# Step 38: Full-Mix Rank-2 LoRA Regularization (Mixed Result)

Step 37 menunjukkan rank-4 alpha-8 memperbaiki retensi real test dibanding
rank-8. Step ini menurunkan kapasitas lagi ke rank-2 alpha-4 (rasio alpha/rank
tetap 2) untuk melihat apakah regresi real bisa ditekan lebih jauh.

## Training

```bash
uv run banking-asr-finetune-mlx-whisper \
  --manifest-path data/training/combined_train_manifest_10h_80_20_candidate.jsonl \
  --output-dir models/mlx-whisper-large-v3-fullmix-200step-lora-last8-r2-a4-lr1e-4-merged \
  --summary-path artifacts/mlx_whisper_large_v3_fullmix_200step_lora_last8_r2_a4_lr1e-4_merged_summary.jsonl \
  --model-name mlx-community/whisper-large-v3-mlx \
  --max-steps 200 \
  --train-scope decoder_last_8_lora \
  --optimizer adamw \
  --learning-rate 1e-4 \
  --lora-rank 2 \
  --lora-alpha 4
```

Summary: `rows_seen=3839`, `completed_steps=200`, `first_loss=0.789`,
`last_loss=0.0141`, `lora_rank=2`, `lora_alpha=4`, `lora_modules=32`.

## Evaluation

```bash
P=artifacts/mlx_whisper_large_v3_fullmix_200step_lora_last8_r2_a4_lr1e-4_merged
M=models/mlx-whisper-large-v3-fullmix-200step-lora-last8-r2-a4-lr1e-4-merged
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

Raw WER / entity-error (%). Columns: baseline / rank-4 Step 37 / rank-2 this.

| Split | Slice | Base WER | S37 WER | S38 WER | Base Ent | S37 Ent | S38 Ent |
|---|---|---|---|---|---|---|---|
| val | all | 12.11 | 13.19 | 13.36 | 17.27 | 7.91 | 10.07 |
| val | banking | 9.69 | 5.91 | 7.06 | 17.27 | 7.91 | 10.07 |
| val | real | 12.51 | 14.39 | 14.39 | 0.00 | 0.00 | 0.00 |
| test | all | 13.02 | 12.33 | 12.86 | 13.96 | 7.92 | 7.71 |
| test | banking | 14.23 | 6.08 | 6.15 | 13.96 | 7.92 | 7.71 |
| test | real | 12.80 | 13.46 | 14.06 | 0.00 | 0.00 | 0.00 |

Postprocessed banking metrics (%):

| Split | Base WER | S37 WER | S38 WER | Base Ent | S37 Ent | S38 Ent |
|---|---|---|---|---|---|---|
| val banking | 4.19 | 4.93 | 4.68 | 1.44 | 5.04 | 3.84 |
| test banking | 10.30 | 5.08 | 4.79 | 3.33 | 5.00 | 3.75 |

Observasi:
- Rank-2 tidak memperbaiki real dibanding rank-4: validation sama (14.39), test memburuk (13.46 -> 14.06).
- Banking raw memburuk terutama validation (5.91 -> 7.06), tetapi test banking hampir sama (6.08 -> 6.15).
- Postprocessed banking membaik vs rank-4 (val WER 4.68, test WER 4.79; entity 3.84/3.75), tapi raw entity dan real test melemah.
- Rank terlalu rendah mengurangi kapasitas adaptasi yang berguna tanpa menyelesaikan regresi real.

## Decision

```
Rank-2 TIDAK menggantikan rank-4. Rank-4 tetap titik terbaik untuk trade-off
banking vs real. Namun rank-2 memberi petunjuk bahwa entity postprocessed bisa
membaik saat LoRA lebih lemah.

Next (Step 39): pertahankan rank-4, turunkan alpha dari 8 ke 4 (`r4/a4`) untuk
mengurangi kekuatan adapter tanpa kehilangan kapasitas rank. Target: real lebih
baik dari Step 37, banking postprocessed minimal setara Step 38.
```
