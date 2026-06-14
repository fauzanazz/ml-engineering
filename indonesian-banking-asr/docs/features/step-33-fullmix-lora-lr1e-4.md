---
title: "Step 33: Full-Mix LoRA lr=1e-4 (Non-Banking Retention Test)"
type: [feature-note, training, evaluation, regression-analysis]
created: 2026-05-17
status: completed
categories: [asr, whisper, mlx, lora, data-mixing]
related:
  - step-32-banking-only-lr1e-4-full-split-eval.md
  - step-31-banking-only-lora-lr1e-4.md
author: fauzan
---

# Step 33: Full-Mix LoRA lr=1e-4 (Non-Banking Retention Test)

Step 32 menunjukkan banking-only LoRA lr=1e-4 menaikkan akurasi banking tapi
meregresi audio non-banking (real) ~2 poin WER. Hipotesis: melatih LoRA pada
**full 10h mix** (2831 babelspeech non-banking + 1008 banking) alih-alih
banking-only akan memulihkan retensi non-banking sambil mempertahankan gain
banking.

## Training

```bash
uv run banking-asr-finetune-mlx-whisper \
  --manifest-path data/training/combined_train_manifest_10h_80_20_candidate.jsonl \
  --output-dir models/mlx-whisper-large-v3-fullmix-200step-lora-last8-r8-lr1e-4-merged \
  --summary-path artifacts/mlx_whisper_large_v3_fullmix_200step_lora_last8_r8_lr1e-4_merged_summary.jsonl \
  --model-name mlx-community/whisper-large-v3-mlx \
  --max-steps 200 \
  --train-scope decoder_last_8_lora \
  --optimizer adamw \
  --learning-rate 1e-4
```

Summary: `rows_seen=3839`, `completed_steps=200`, `first_loss=0.789`,
`last_loss=0.0074`, `lora_rank=8`, `lora_alpha=16`, `lora_modules=32`.

## Evaluation

```bash
P=artifacts/mlx_whisper_large_v3_fullmix_200step_lora_last8_r8_lr1e-4_merged
M=models/mlx-whisper-large-v3-fullmix-200step-lora-last8-r8-lr1e-4-merged
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

Raw, WER / entity-error (%). Columns: baseline / banking-only(Step 32) / full-mix(this).

| Split | Slice | Base WER | S32 WER | S33 WER | Base Ent | S32 Ent | S33 Ent |
|---|---|---|---|---|---|---|---|
| val | all | 12.11 | 13.72 | 13.19 | 17.27 | 12.23 | 6.47 |
| val | banking | 9.69 | 8.21 | 5.58 | 17.27 | 12.23 | 6.47 |
| val | real | 12.51 | 14.62 | 14.45 | 0.00 | 0.00 | 0.00 |
| test | all | 13.02 | 12.84 | 12.75 | 13.96 | 9.17 | 7.08 |
| test | banking | 14.23 | 7.87 | 5.87 | 13.96 | 9.17 | 7.08 |
| test | real | 12.80 | 13.74 | 13.98 | 0.00 | 0.00 | 0.00 |

Postprocessed, WER / entity-error (%):

| Split | Slice | Base WER | S32 WER | S33 WER | Base Ent | S32 Ent | S33 Ent |
|---|---|---|---|---|---|---|---|
| val | banking | 4.19 | 5.91 | 5.09 | 1.44 | 6.24 | 5.04 |
| test | banking | 10.30 | 6.51 | 5.01 | 3.33 | 5.21 | 4.58 |
| val | real | 12.51 | 14.62 | 14.45 | 0.00 | 0.00 | 0.00 |
| test | real | 12.81 | 13.75 | 13.98 | 0.00 | 0.00 | 0.00 |

Observasi:
- Full-mix lebih baik dari banking-only pada banking: raw WER val 8.21 -> 5.58, test 7.87 -> 5.87; entity raw turun lagi (val 12.23 -> 6.47, test 9.17 -> 7.08). Entity terbaik sejauh ini.
- Retensi non-banking TIDAK pulih: real WER val 12.51 -> 14.45, test 12.80 -> 13.98, nyaris sama dengan regresi banking-only. Mencampur data non-banking ke train tidak menghilangkan regresi.
- Regresi real bukan dari komposisi data tapi dari distorsi decoder oleh LoRA (200 step, lr=1e-4, decoder_last_8). Penyebab dugaan: gaya/format label train berbeda dari distribusi referensi real eval.

## Decision

```
Full-mix > banking-only untuk banking dan entity, jadikan default arah training.
Regresi non-banking ~1.5-2 poin WER persisten dan TIDAK disebabkan komposisi data
=> sumbernya kapasitas/durasi adaptasi LoRA (overfit gaya), bukan data mix.

Next (Step 34): turunkan agresivitas adaptasi pada full-mix untuk menekan regresi
real sambil menjaga gain banking -- sweep langkah lebih pendek / lr lebih rendah
(mis. 100 step lr=1e-4 atau 200 step lr=5e-5) dan ukur kembali slice real vs
banking. Target: real WER kembali ~baseline (<=12.6) dengan banking tetap <7.
```
