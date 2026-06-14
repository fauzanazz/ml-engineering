---
title: "Step 34: Full-Mix 100-Step Probe (Negative Result)"
type: [feature-note, training, evaluation, regression-analysis, negative-result]
created: 2026-05-17
status: completed
categories: [asr, whisper, mlx, lora, ablation]
related:
  - step-33-fullmix-lora-lr1e-4.md
  - step-32-banking-only-lr1e-4-full-split-eval.md
author: fauzan
---

# Step 34: Full-Mix 100-Step Probe (Negative Result)

Hipotesis dari Step 33: regresi non-banking adalah overfit gaya dari adaptasi
LoRA yang terlalu lama, sehingga memendekkan training (200 -> 100 step) pada
full-mix lr=1e-4 seharusnya menekan regresi real sambil menjaga gain banking.

## Training

```bash
uv run banking-asr-finetune-mlx-whisper \
  --manifest-path data/training/combined_train_manifest_10h_80_20_candidate.jsonl \
  --output-dir models/mlx-whisper-large-v3-fullmix-100step-lora-last8-r8-lr1e-4-merged \
  --summary-path artifacts/mlx_whisper_large_v3_fullmix_100step_lora_last8_r8_lr1e-4_merged_summary.jsonl \
  --model-name mlx-community/whisper-large-v3-mlx \
  --max-steps 100 --train-scope decoder_last_8_lora --optimizer adamw --learning-rate 1e-4
```

Summary: `completed_steps=100`, `first_loss=0.789`, `last_loss=0.0495`
(vs 200-step `last_loss=0.0074`).

## Evaluation

Transcribe + postprocess + eval all/banking/real for validation & test (identik
pipeline Step 33).

## Results

Raw WER / entity-error (%). Columns: baseline / full-mix 200-step (Step 33) / full-mix 100-step (this).

| Split | Slice | Base WER | S33 WER | S34 WER | Base Ent | S33 Ent | S34 Ent |
|---|---|---|---|---|---|---|---|
| val | all | 12.11 | 13.19 | 14.90 | 17.27 | 6.47 | 7.19 |
| val | banking | 9.69 | 5.58 | 5.67 | 17.27 | 6.47 | 7.19 |
| val | real | 12.51 | 14.45 | 16.43 | 0.00 | 0.00 | 0.00 |
| test | all | 13.02 | 12.75 | 13.30 | 13.96 | 7.08 | 8.75 |
| test | banking | 14.23 | 5.87 | 6.72 | 13.96 | 7.08 | 8.75 |
| test | real | 12.80 | 13.98 | 14.48 | 0.00 | 0.00 | 0.00 |

Observasi:
- Memendekkan ke 100 step JUSTRU memperburuk semuanya: real val 14.45 -> 16.43, test 13.98 -> 14.48; banking juga turun (val 5.58 -> 5.67, test 5.87 -> 6.72) dan entity naik.
- 200-step (last_loss 0.0074) generalisasi lebih baik dari 100-step (last_loss 0.0495). Step count lebih sedikit BUKAN obat regresi.
- Kesimpulan: regresi non-banking bukan murni overfit durasi. Penyebab lebih mungkin: distorsi decoder oleh adaptasi `decoder_last_8` pada lr=1e-4 (kapasitas/cakupan modul terlalu besar), bukan jumlah langkah.

## Decision

```
Hipotesis "kurangi step" DITOLAK -- 100 step lebih buruk dari 200 step di semua
slice. Lever yang benar bukan durasi tapi besar/cakupan perubahan decoder.

Next (Step 35): kurangi cakupan/kapasitas adaptasi pada full-mix 200-step --
turunkan learning rate (lr=5e-5) ATAU perkecil scope ke decoder_last_4_lora,
agar decoder non-banking lebih terjaga sambil banking tetap membaik. Tetap
basis full-mix 200-step (konfig terbaik sejauh ini dari Step 33).
```
