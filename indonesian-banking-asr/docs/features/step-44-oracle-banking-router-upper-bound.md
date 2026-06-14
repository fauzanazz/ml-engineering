---
title: "Step 44: Oracle Banking Router Upper Bound"
type: [feature-note, evaluation, routing, upper-bound]
created: 2026-05-17
status: completed
categories: [asr, whisper, routing, model-selection]
related:
  - step-43-v2-candidate-selection-matrix.md
  - step-42-postprocess-v2-implementation.md
author: fauzan
---

# Step 44: Oracle Banking Router Upper Bound

Step 43 showed no single model dominates: baseline+v2 preserves real/validation,
Step37+v2 wins test banking. Step ini evaluates an oracle router upper bound:
use Step37+v2 for banking synthetic rows and baseline+v2 for real rows.

## Setup

Oracle policy:

```text
if utterance_id in combined_10h_<split>_banking_manifest.jsonl:
    use Step37 r4/a8 + postprocess v2 prediction
else:
    use baseline + postprocess v2 prediction
```

Output prefix:

```text
artifacts/mlx_whisper_large_v3_oracle_banking_s37_real_baseline_v2
```

## Commands

```bash
python3 - <<'PY'
# Merge baseline_v2 and Step37_v2 predictions by manifest oracle source.
PY

P=artifacts/mlx_whisper_large_v3_oracle_banking_s37_real_baseline_v2
for split in validation test; do
  for slice in "" "_banking" "_real"; do
    uv run banking-asr-evaluate \
      --manifest-path artifacts/combined_10h_${split}${slice}_manifest.jsonl \
      --predictions-path ${P}_${split}_postprocessed_predictions.jsonl \
      --output-path ${P}_${split}${slice}_postprocessed_eval.jsonl
  done
done
```

## Results

v2 postprocessed WER / entity-error (%):

| Split | Pipeline | All | Banking | Real |
|---|---|---|---|---|
| val | baseline+v2 | 11.30 / 0.72 | 3.94 / 0.72 | 12.51 / 0.00 |
| val | Step37+v2 | 12.95 / 2.88 | 4.19 / 2.88 | 14.39 / 0.00 |
| val | oracle route | 11.33 / 2.88 | 4.19 / 2.88 | 12.51 / 0.00 |
| test | baseline+v2 | 12.35 / 1.88 | 9.80 / 1.88 | 12.81 / 0.00 |
| test | Step37+v2 | 11.97 / 1.67 | 3.72 / 1.67 | 13.46 / 0.00 |
| test | oracle route | 11.43 / 1.67 | 3.72 / 1.67 | 12.81 / 0.00 |

## Observations

- Oracle route is best test pipeline: all WER 11.43, banking WER 3.72, real WER retained at baseline 12.81.
- Oracle route resolves Step37's real regression by using baseline on real rows.
- Validation oracle is close to baseline all WER (11.33 vs 11.30) but worse entity (2.88 vs 0.72) because validation banking baseline is already strong.
- Test split benefits strongly from routing because baseline test banking is weak (9.80 WER), while Step37 test banking is strong (3.72 WER).
- Routing value is split-dependent but material on target test banking.

## Decision

```
Oracle routing upper bound is ACCEPTED as strategy direction.
Best deployment shape is not single model; it is domain-aware routing:
- banking domain -> Step37 model + postprocess v2
- general/non-banking -> baseline + postprocess v2

Next (Step 45): test a non-oracle heuristic router based on transcript/domain
keywords to estimate how much of the oracle gain is reachable without manifest
source labels.
```
