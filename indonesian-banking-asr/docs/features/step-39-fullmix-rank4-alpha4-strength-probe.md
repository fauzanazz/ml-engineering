---
title: "Step 39: Full-Mix Rank-4 Alpha-4 Strength Probe (Negative Result)"
type: [feature-note, training, evaluation, regression-analysis, negative-result]
created: 2026-05-17
status: completed
categories: [asr, whisper, mlx, lora, alpha-ablation]
related:
  - step-38-fullmix-rank2-lora-regularization.md
  - step-37-fullmix-rank4-lora-regularization.md
author: fauzan
---

# Step 39: Full-Mix Rank-4 Alpha-4 Strength Probe (Negative Result)

Step 37 menemukan rank-4 alpha-8 sebagai trade-off terbaik sementara. Step 38
menunjukkan rank-2 terlalu lemah. Step ini mempertahankan rank-4 tetapi
menurunkan alpha dari 8 ke 4 untuk mengurangi kekuatan adapter tanpa menurunkan
rank.

## Training

```bash
uv run banking-asr-finetune-mlx-whisper \
  --manifest-path data/training/combined_train_manifest_10h_80_20_candidate.jsonl \
  --output-dir models/mlx-whisper-large-v3-fullmix-200step-lora-last8-r4-a4-lr1e-4-merged \
  --summary-path artifacts/mlx_whisper_large_v3_fullmix_200step_lora_last8_r4_a4_lr1e-4_merged_summary.jsonl \
  --model-name mlx-community/whisper-large-v3-mlx \
  --max-steps 200 \
  --train-scope decoder_last_8_lora \
  --optimizer adamw \
  --learning-rate 1e-4 \
  --lora-rank 4 \
  --lora-alpha 4
```

Summary: `rows_seen=3839`, `completed_steps=200`, `first_loss=0.789`,
`last_loss=0.0159`, `lora_rank=4`, `lora_alpha=4`, `lora_modules=32`.

## Evaluation

```bash
P=artifacts/mlx_whisper_large_v3_fullmix_200step_lora_last8_r4_a4_lr1e-4_merged
M=models/mlx-whisper-large-v3-fullmix-200step-lora-last8-r4-a4-lr1e-4-merged
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

Raw WER / entity-error (%). Columns: baseline / rank-4 alpha-8 Step 37 / rank-4 alpha-4 this.

| Split | Slice | Base WER | S37 WER | S39 WER | Base Ent | S37 Ent | S39 Ent |
|---|---|---|---|---|---|---|---|
| val | all | 12.11 | 13.19 | 13.68 | 17.27 | 7.91 | 8.87 |
| val | banking | 9.69 | 5.91 | 6.40 | 17.27 | 7.91 | 8.87 |
| val | real | 12.51 | 14.39 | 14.88 | 0.00 | 0.00 | 0.00 |
| test | all | 13.02 | 12.33 | 13.33 | 13.96 | 7.92 | 7.50 |
| test | banking | 14.23 | 6.08 | 6.65 | 13.96 | 7.92 | 7.50 |
| test | real | 12.80 | 13.46 | 14.54 | 0.00 | 0.00 | 0.00 |

Postprocessed banking metrics (%):

| Split | Base WER | S37 WER | S39 WER | Base Ent | S37 Ent | S39 Ent |
|---|---|---|---|---|---|---|
| val banking | 4.19 | 4.93 | 4.68 | 1.44 | 5.04 | 4.08 |
| test banking | 10.30 | 5.08 | 5.58 | 3.33 | 5.00 | 4.38 |

Observasi:
- Alpha-4 memperburuk real dibanding alpha-8: val 14.39 -> 14.88, test 13.46 -> 14.54.
- Banking raw juga turun: val 5.91 -> 6.40, test 6.08 -> 6.65.
- Entity test raw sedikit membaik (7.92 -> 7.50), tetapi WER trade-off buruk.
- Rank-4 alpha-8 tetap konfigurasi regularized terbaik; melemahkan alpha membuat model kurang adaptif dan tidak menjaga real.

## Decision

```
Hipotesis "rank-4 alpha lebih kecil memperbaiki retensi" DITOLAK.
Best training remains Step 37: full-mix 200-step last-8 r4/a8 lr=1e-4.

Next (Step 40): berhenti menambah training sweep untuk sementara; audit error
postprocessing/entity pada kandidat terbaik (Step 37) vs baseline dan Step 33.
Fokus cari perbaikan tanpa melatih ulang, karena training levers belum membuat
real kembali baseline.
```
