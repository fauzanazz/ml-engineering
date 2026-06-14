---
title: "Step 35: Full-Mix lr=5e-5 Probe (Negative Result)"
type: [feature-note, training, evaluation, regression-analysis, negative-result]
created: 2026-05-17
status: completed
categories: [asr, whisper, mlx, lora, learning-rate]
related:
  - step-34-fullmix-100step-regression-probe.md
  - step-33-fullmix-lora-lr1e-4.md
author: fauzan
---

# Step 35: Full-Mix lr=5e-5 Probe (Negative Result)

Step 34 menolak hipotesis bahwa durasi training lebih pendek memperbaiki regresi
non-banking. Step ini menguji lever lebih halus: tetap full-mix 200-step dan
`decoder_last_8_lora`, tapi turunkan learning rate dari `1e-4` ke `5e-5` untuk
mengurangi distorsi decoder.

## Training

```bash
uv run banking-asr-finetune-mlx-whisper \
  --manifest-path data/training/combined_train_manifest_10h_80_20_candidate.jsonl \
  --output-dir models/mlx-whisper-large-v3-fullmix-200step-lora-last8-r8-lr5e-5-merged \
  --summary-path artifacts/mlx_whisper_large_v3_fullmix_200step_lora_last8_r8_lr5e-5_merged_summary.jsonl \
  --model-name mlx-community/whisper-large-v3-mlx \
  --max-steps 200 \
  --train-scope decoder_last_8_lora \
  --optimizer adamw \
  --learning-rate 5e-5
```

Summary: `rows_seen=3839`, `completed_steps=200`, `first_loss=0.789`,
`last_loss=0.0144`, `lora_rank=8`, `lora_alpha=16`, `lora_modules=32`.

## Evaluation

```bash
P=artifacts/mlx_whisper_large_v3_fullmix_200step_lora_last8_r8_lr5e-5_merged
M=models/mlx-whisper-large-v3-fullmix-200step-lora-last8-r8-lr5e-5-merged
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

Raw WER / entity-error (%). Columns: baseline / full-mix lr=1e-4 (Step 33) / full-mix lr=5e-5 (this).

| Split | Slice | Base WER | S33 WER | S35 WER | Base Ent | S33 Ent | S35 Ent |
|---|---|---|---|---|---|---|---|
| val | all | 12.11 | 13.19 | 13.69 | 17.27 | 6.47 | 9.59 |
| val | banking | 9.69 | 5.58 | 6.81 | 17.27 | 6.47 | 9.59 |
| val | real | 12.51 | 14.45 | 14.83 | 0.00 | 0.00 | 0.00 |
| test | all | 13.02 | 12.75 | 12.88 | 13.96 | 7.08 | 7.92 |
| test | banking | 14.23 | 5.87 | 6.22 | 13.96 | 7.08 | 7.92 |
| test | real | 12.80 | 13.98 | 14.07 | 0.00 | 0.00 | 0.00 |

Postprocessed banking metrics (%):

| Split | Base WER | S33 WER | S35 WER | Base Ent | S33 Ent | S35 Ent |
|---|---|---|---|---|---|---|
| val banking | 4.19 | 5.09 | 4.60 | 1.44 | 5.04 | 3.84 |
| test banking | 10.30 | 5.01 | 5.01 | 3.33 | 4.58 | 4.38 |

Observasi:
- Menurunkan lr ke `5e-5` tidak memperbaiki non-banking real: val real 14.45 -> 14.83, test real 13.98 -> 14.07 (lebih buruk dari Step 33 dan baseline).
- Banking raw juga turun: val WER 5.58 -> 6.81, test WER 5.87 -> 6.22; entity raw naik 6.47/7.08 -> 9.59/7.92.
- Postprocessed banking sedikit membaik di validation (WER 5.09 -> 4.60, Ent 5.04 -> 3.84), tapi raw signal melemah dan real tetap regresi.

## Decision

```
Hipotesis "turunkan lr ke 5e-5 untuk retensi" DITOLAK.
Lr=1e-4 tetap lebih baik untuk raw banking dan tidak lebih buruk pada real.
Regresi real belum terselesaikan oleh durasi lebih pendek atau lr lebih rendah.

Next (Step 36): ubah scope, bukan lr/step -- full-mix 200-step lr=1e-4 dengan
`decoder_last_4_lora`. Target: kurangi jumlah modul yang diadaptasi dari 32 ke
16 agar real WER turun, sambil banking tetap mendekati Step 33.
```
