---
title: "Step 36: Full-Mix last-4 LoRA Scope Probe (Negative Result)"
type: [feature-note, training, evaluation, regression-analysis, negative-result]
created: 2026-05-17
status: completed
categories: [asr, whisper, mlx, lora, scope-ablation]
related:
  - step-35-fullmix-lr5e-5-regression-probe.md
  - step-33-fullmix-lora-lr1e-4.md
author: fauzan
---

# Step 36: Full-Mix last-4 LoRA Scope Probe (Negative Result)

Step 35 menolak penurunan learning rate sebagai obat regresi real. Step ini
mengubah scope adaptasi: full-mix 200-step lr=1e-4 tetap sama, tapi LoRA hanya
pada `decoder_last_4_lora` (16 module) alih-alih `decoder_last_8_lora` (32
module). Hipotesis: scope lebih kecil menjaga non-banking real.

## Training

```bash
uv run banking-asr-finetune-mlx-whisper \
  --manifest-path data/training/combined_train_manifest_10h_80_20_candidate.jsonl \
  --output-dir models/mlx-whisper-large-v3-fullmix-200step-lora-last4-r8-lr1e-4-merged \
  --summary-path artifacts/mlx_whisper_large_v3_fullmix_200step_lora_last4_r8_lr1e-4_merged_summary.jsonl \
  --model-name mlx-community/whisper-large-v3-mlx \
  --max-steps 200 \
  --train-scope decoder_last_4_lora \
  --optimizer adamw \
  --learning-rate 1e-4
```

Summary: `rows_seen=3839`, `completed_steps=200`, `first_loss=0.789`,
`last_loss=0.0375`, `lora_rank=8`, `lora_alpha=16`, `lora_modules=16`.

## Evaluation

```bash
P=artifacts/mlx_whisper_large_v3_fullmix_200step_lora_last4_r8_lr1e-4_merged
M=models/mlx-whisper-large-v3-fullmix-200step-lora-last4-r8-lr1e-4-merged
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

Raw WER / entity-error (%). Columns: baseline / last-8 (Step 33) / last-4 (this).

| Split | Slice | Base WER | S33 WER | S36 WER | Base Ent | S33 Ent | S36 Ent |
|---|---|---|---|---|---|---|---|
| val | all | 12.11 | 13.19 | 15.40 | 17.27 | 6.47 | 7.67 |
| val | banking | 9.69 | 5.58 | 6.73 | 17.27 | 6.47 | 7.67 |
| val | real | 12.51 | 14.45 | 16.83 | 0.00 | 0.00 | 0.00 |
| test | all | 13.02 | 12.75 | 16.19 | 13.96 | 7.08 | 6.04 |
| test | banking | 14.23 | 5.87 | 8.30 | 13.96 | 7.08 | 6.04 |
| test | real | 12.80 | 13.98 | 17.61 | 0.00 | 0.00 | 0.00 |

Postprocessed banking metrics (%):

| Split | Base WER | S33 WER | S36 WER | Base Ent | S33 Ent | S36 Ent |
|---|---|---|---|---|---|---|
| val banking | 4.19 | 5.09 | 5.09 | 1.44 | 5.04 | 2.88 |
| test banking | 10.30 | 5.01 | 7.65 | 3.33 | 4.58 | 3.96 |

Observasi:
- `decoder_last_4_lora` gagal menjaga real; real WER memburuk tajam: val 14.45 -> 16.83, test 13.98 -> 17.61.
- Banking raw juga turun dari Step 33: val WER 5.58 -> 6.73, test 5.87 -> 8.30.
- Entity postprocessed sedikit lebih baik dari Step 33, tapi trade-off WER terlalu buruk.
- Scope lebih kecil bukan regularisasi baik; kemungkinan underfit/instabil karena hanya layer paling akhir berubah, membuat distribusi decoder lebih tajam tapi kurang robust.

## Decision

```
Hipotesis "scope last-4 memperbaiki retensi" DITOLAK.
Best training tetap full-mix 200-step `decoder_last_8_lora` lr=1e-4 dari Step 33.

Next (Step 37): pertahankan last-8 lr=1e-4, tetapi kurangi kapasitas LoRA via
rank lebih rendah (`--lora-rank 4`, alpha default/proporsional) untuk melihat
apakah low-rank regularization menekan regresi real tanpa kehilangan banking.
```
