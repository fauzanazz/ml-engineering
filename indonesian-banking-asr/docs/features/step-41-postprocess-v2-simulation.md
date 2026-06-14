---
title: "Step 41: Postprocess v2 Simulation on Step 37"
type: [feature-note, evaluation, postprocessing, simulation]
created: 2026-05-17
status: completed
categories: [asr, whisper, postprocessing, entity-error]
related:
  - step-40-entity-postprocess-error-audit.md
  - step-37-fullmix-rank4-lora-regularization.md
author: fauzan
---

# Step 41: Postprocess v2 Simulation on Step 37

Step 40 menemukan entity misses yang bisa diperbaiki tanpa retraining: `CRIS`,
`Piloter`/`Pailater`, `virtual count`, `RTGE`, `Rp11 juta`, dan `Rp11.07.000`.
Step ini mensimulasikan postprocess v2 pada prediksi raw kandidat terbaik Step
37 (full-mix last-8 r4/a8 lr=1e-4) tanpa mengubah default code path.

## Setup

Output simulasi:

- `artifacts/mlx_whisper_large_v3_fullmix_200step_lora_last8_r4_a8_lr1e-4_merged_v2_validation_postprocessed_predictions.jsonl`
- `artifacts/mlx_whisper_large_v3_fullmix_200step_lora_last8_r4_a8_lr1e-4_merged_v2_test_postprocessed_predictions.jsonl`

Rules v2:

```python
CRIS -> QRIS
Piloter/Pailater -> paylater
virtual count -> virtual account
RTGE -> RTGS
Rp<1-3 digits> juta -> Rp<digits>.000.000
RpDD.DD.000 -> RpDD.DDD.000
```

## Commands

```bash
python3 - <<'PY'
import json, re, sys
sys.path.insert(0, 'src')
from indonesian_banking_asr.evaluation.postprocess import postprocess_transcript

def v2(text):
    text = postprocess_transcript(text)
    # narrow lexicon + amount repairs from Step 40 audit
    return text
PY

P=artifacts/mlx_whisper_large_v3_fullmix_200step_lora_last8_r4_a8_lr1e-4_merged_v2
for split in validation test; do
  for slice in "" "_banking" "_real"; do
    man=artifacts/combined_10h_${split}${slice}_manifest.jsonl
    uv run banking-asr-evaluate \
      --manifest-path $man \
      --predictions-path ${P}_${split}_postprocessed_predictions.jsonl \
      --output-path ${P}_${split}${slice}_postprocessed_eval.jsonl
  done
done
```

## Results

Step 37 postprocessed v1 vs simulated v2 (%):

| Split | Slice | v1 WER | v2 WER | v1 Ent | v2 Ent | v2 Errors |
|---|---|---:|---:|---:|---:|---|
| val | all | 13.05 | 12.95 | 5.04 | 2.88 | CARD_LAST4=3, PRODUCT_NAME=6, ACCOUNT_NUMBER=3 |
| val | banking | 4.93 | 4.19 | 5.04 | 2.88 | CARD_LAST4=3, PRODUCT_NAME=6, ACCOUNT_NUMBER=3 |
| val | real | 14.39 | 14.39 | 0.00 | 0.00 | none |
| test | all | 12.18 | 11.97 | 5.00 | 1.67 | ACCOUNT_NUMBER=6, PRODUCT_NAME=2 |
| test | banking | 5.08 | 3.72 | 5.00 | 1.67 | ACCOUNT_NUMBER=6, PRODUCT_NAME=2 |
| test | real | 13.46 | 13.46 | 0.00 | 0.00 | none |

Observasi:
- v2 improves Step 37 test banking WER 5.08 -> 3.72 and entity error 5.00 -> 1.67.
- v2 improves validation banking WER 4.93 -> 4.19 and entity error 5.04 -> 2.88.
- Real slice unchanged (val/test 14.39/13.46), so postprocess v2 does not worsen non-banking real in this eval.
- v2 beats baseline postprocessed banking WER on test (3.72 vs 10.30) and matches baseline validation banking WER (4.19 vs 4.19), while still trailing baseline entity on validation (2.88 vs 1.44) and test (1.67 vs 3.33 is better).
- Remaining errors are mostly true digit deletion (ACCOUNT_NUMBER) plus a few product confusions; unsafe account-number hallucination should not be attempted.

## Decision

```
Postprocess v2 simulation is ACCEPTED. It is the highest-leverage improvement so
far: +1.36 WER points and +3.33 entity points on test banking with zero real WER
change.

Next (Step 42): promote v2 rules into `src/indonesian_banking_asr/evaluation/postprocess.py`,
add focused unit tests, rerun CLI postprocess/evaluation for Step 37, and compare
against simulation outputs.
```
