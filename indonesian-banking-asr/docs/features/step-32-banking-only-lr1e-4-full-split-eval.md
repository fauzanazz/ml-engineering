---
title: "Step 32: Banking-Only LoRA lr=1e-4 Full-Split Evaluation"
type: [feature-note, evaluation, regression-analysis]
created: 2026-05-17
status: completed
categories: [asr, whisper, mlx, lora, evaluation]
related:
  - step-31-banking-only-lora-lr1e-4.md
  - step-30-banking-only-lora-lr5e-5.md
author: fauzan
---

# Step 32: Banking-Only LoRA lr=1e-4 Full-Split Evaluation

Step 31 hanya diuji pada smoke slice 20 baris. Step ini mengevaluasi model terbaik
(`models/mlx-whisper-large-v3-banking-only-200step-lora-last8-r8-lr1e-4-merged`)
pada full validation (485) dan test (511) split, dipecah ke slice all / banking /
real, untuk mengukur perbaikan banking sekaligus mendeteksi regresi pada audio
non-banking (catastrophic forgetting).

## Evaluation

Transcribe full split:

```bash
M=models/mlx-whisper-large-v3-banking-only-200step-lora-last8-r8-lr1e-4-merged
P=artifacts/mlx_whisper_large_v3_banking_only_200step_lora_last8_r8_lr1e-4_merged
uv run banking-asr-transcribe-whisper --manifest-path artifacts/combined_10h_validation_manifest.jsonl --model $M --output-path ${P}_validation_predictions.jsonl --language id
uv run banking-asr-transcribe-whisper --manifest-path artifacts/combined_10h_test_manifest.jsonl --model $M --output-path ${P}_test_predictions.jsonl --language id
```

Postprocess + evaluate semua slice (all / banking / real, raw + postprocessed):

```bash
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

Raw WER / entity-error (%), model vs baseline.

| Split | Slice | Rows | Baseline WER | Model WER | Baseline Ent | Model Ent |
|---|---|---|---|---|---|---|
| val | all | 485 | 12.11 | 13.72 | 17.27 | 12.23 |
| val | banking | 132 | 9.69 | 8.21 | 17.27 | 12.23 |
| val | real | 353 | 12.51 | 14.62 | 0.00 | 0.00 |
| test | all | 511 | 13.02 | 12.84 | 13.96 | 9.17 |
| test | banking | 156 | 14.23 | 7.87 | 13.96 | 9.17 |
| test | real | 355 | 12.80 | 13.74 | 0.00 | 0.00 |

Postprocessed WER / entity-error (%):

| Split | Slice | Baseline WER | Model WER | Baseline Ent | Model Ent |
|---|---|---|---|---|---|
| val | all | 11.33 | 13.39 | 1.44 | 6.24 |
| val | banking | 4.19 | 5.91 | 1.44 | 6.24 |
| val | real | 12.51 | 14.62 | 0.00 | 0.00 |
| test | all | 12.43 | 12.65 | 3.33 | 5.21 |
| test | banking | 10.30 | 6.51 | 3.33 | 5.21 |
| test | real | 12.81 | 13.75 | 0.00 | 0.00 |

Observasi:
- Banking raw membaik kuat: test WER 14.23 -> 7.87, val 9.69 -> 8.21; entity raw turun (test 13.96 -> 9.17, val 17.27 -> 12.23).
- Non-banking real regresi: val WER 12.51 -> 14.62, test 12.80 -> 13.74. Sinyal catastrophic forgetting dari banking-only LoRA.
- Postprocessed banking entity justru naik (val 1.44 -> 6.24, test 3.33 -> 5.21). Output model bergeser dari distribusi yang diasumsikan normalizer postproc; gain raw tidak sepenuhnya diteruskan setelah postproc.

## Decision

```
Banking-only LoRA lr=1e-4 menghasilkan gain banking nyata pada raw output tapi:
(1) meregresi non-banking real ~2 poin WER, dan
(2) postproc entity malah memburuk (mismatch normalizer).

Next (Step 33): latih LoRA pada full 10h mix (bukan banking-only) dengan lr=1e-4
untuk menguji apakah retensi non-banking pulih sambil mempertahankan gain banking.
Jika regresi real hilang, lanjut sweep step/rank; jika tidak, evaluasi mixing
ratio data. Postproc normalizer audit ditunda ke Step 41-45.
```
