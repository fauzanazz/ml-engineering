---
title: "Step 43: v2 Candidate Selection Matrix"
type: [feature-note, evaluation, model-selection]
created: 2026-05-17
status: completed
categories: [asr, whisper, postprocessing, model-selection]
related:
  - step-42-postprocess-v2-implementation.md
  - step-37-fullmix-rank4-lora-regularization.md
  - step-33-fullmix-lora-lr1e-4.md
author: fauzan
---

# Step 43: v2 Candidate Selection Matrix

Step 42 promoted postprocess v2. Step ini reprocess raw predictions from top
candidates with the same v2 code path so model selection is not biased by older
postprocess v1 outputs.

## Candidates

- `baseline`: `artifacts/mlx_whisper_large_v3_baseline_combined_10h`
- `s33_r8a16`: full-mix last-8 rank-8 alpha-16 lr=1e-4
- `s37_r4a8`: full-mix last-8 rank-4 alpha-8 lr=1e-4
- `s38_r2a4`: full-mix last-8 rank-2 alpha-4 lr=1e-4

## Commands

```bash
for candidate in baseline s33_r8a16 s37_r4a8 s38_r2a4; do
  for split in validation test; do
    uv run banking-asr-postprocess-predictions \
      --predictions-path ${PREFIX}_${split}_predictions.jsonl \
      --output-path ${PREFIX}_v2matrix_${split}_postprocessed_predictions.jsonl
    for slice in "" "_banking" "_real"; do
      uv run banking-asr-evaluate \
        --manifest-path artifacts/combined_10h_${split}${slice}_manifest.jsonl \
        --predictions-path ${PREFIX}_v2matrix_${split}_postprocessed_predictions.jsonl \
        --output-path ${PREFIX}_v2matrix_${split}${slice}_postprocessed_eval.jsonl
    done
  done
done
```

## Results

v2 postprocessed WER / entity-error (%):

| Split | Candidate | All | Banking | Real |
|---|---|---|---|---|
| val | baseline | 11.30 / 0.72 | 3.94 / 0.72 | 12.51 / 0.00 |
| val | s33_r8a16 | 13.02 / 2.88 | 4.35 / 2.88 | 14.45 / 0.00 |
| val | s37_r4a8 | 12.95 / 2.88 | 4.19 / 2.88 | 14.39 / 0.00 |
| val | s38_r2a4 | 12.98 / 3.12 | 4.43 / 3.12 | 14.39 / 0.00 |
| test | baseline | 12.35 / 1.88 | 9.80 / 1.88 | 12.81 / 0.00 |
| test | s33_r8a16 | 12.43 / 1.67 | 3.79 / 1.67 | 13.98 / 0.00 |
| test | s37_r4a8 | 11.97 / 1.67 | 3.72 / 1.67 | 13.46 / 0.00 |
| test | s38_r2a4 | 12.50 / 1.46 | 3.79 / 1.46 | 14.06 / 0.00 |

## Observations

- Baseline + v2 is best on validation overall and real slice, but fails test banking (9.80 WER vs 3.72 for Step 37).
- Step 37 + v2 is best test pipeline: all WER 11.97, banking WER 3.72, entity 1.67.
- Step 38 has slightly better test entity (1.46) but worse all/real WER than Step 37.
- Fine-tuned candidates still regress real vs baseline; Step 37 has smallest test real regression among tuned candidates (13.46 vs baseline 12.81).
- Validation favors baseline, test favors Step 37; this split mismatch suggests model choice depends on banking-vs-real routing or business priority.

## Decision

```
Best single fine-tuned pipeline: Step 37 model + postprocess v2.
Best non-banking-safe pipeline: baseline + postprocess v2.
No single model dominates both banking and real.

Next (Step 44): evaluate an oracle banking router upper bound: use Step 37+v2
for banking synthetic rows and baseline+v2 for real rows. This quantifies the
value of intent/domain routing before choosing production strategy.
```
