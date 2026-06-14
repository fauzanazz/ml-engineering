---
title: "Step 42: Postprocess v2 Implementation"
type: [feature-note, implementation, evaluation, postprocessing]
created: 2026-05-17
status: completed
categories: [asr, whisper, postprocessing, tests]
related:
  - step-41-postprocess-v2-simulation.md
  - step-40-entity-postprocess-error-audit.md
author: fauzan
---

# Step 42: Postprocess v2 Implementation

Step 41 proved v2 postprocessing improves Step 37 entity and WER without changing
real-slice WER. Step ini promotes rules v2 into production postprocess code and
validates CLI output matches simulation.

## Changes

Files changed:

- `src/indonesian_banking_asr/evaluation/postprocess.py`
- `src/indonesian_banking_asr/evaluation/postprocess_cli.py`
- `tests/evaluation/test_postprocess.py`
- `tests/evaluation/test_postprocess_cli.py`

Rules added:

```python
cris -> QRIS
piloter -> paylater
pailater -> paylater
virtual count -> virtual account
rtge -> RTGS
Rp<1-3 digits> juta -> Rp<digits>.000.000
RpDD.DD.000 -> RpDD.DDD.000
```

CLI `postprocess` label updated from `banking_entity_v1` to `banking_entity_v2`.

## Tests

```bash
uv run pytest tests/evaluation/test_postprocess.py tests/evaluation/test_postprocess_cli.py
```

Result:

```text
4 passed in 0.01s
```

## Evaluation

Regenerated Step 37 postprocessed predictions using CLI v2:

```bash
SRC=artifacts/mlx_whisper_large_v3_fullmix_200step_lora_last8_r4_a8_lr1e-4_merged
OUT=artifacts/mlx_whisper_large_v3_fullmix_200step_lora_last8_r4_a8_lr1e-4_merged_v2cli
for split in validation test; do
  uv run banking-asr-postprocess-predictions \
    --predictions-path ${SRC}_${split}_predictions.jsonl \
    --output-path ${OUT}_${split}_postprocessed_predictions.jsonl
  for slice in "" "_banking" "_real"; do
    man=artifacts/combined_10h_${split}${slice}_manifest.jsonl
    uv run banking-asr-evaluate \
      --manifest-path $man \
      --predictions-path ${OUT}_${split}_postprocessed_predictions.jsonl \
      --output-path ${OUT}_${split}${slice}_postprocessed_eval.jsonl
  done
done
```

CLI v2 exactly matched Step 41 simulation:

| Split | Slice | WER % | Entity % | Remaining Errors |
|---|---|---:|---:|---|
| val | all | 12.95 | 2.88 | CARD_LAST4=3, PRODUCT_NAME=6, ACCOUNT_NUMBER=3 |
| val | banking | 4.19 | 2.88 | CARD_LAST4=3, PRODUCT_NAME=6, ACCOUNT_NUMBER=3 |
| val | real | 14.39 | 0.00 | none |
| test | all | 11.97 | 1.67 | ACCOUNT_NUMBER=6, PRODUCT_NAME=2 |
| test | banking | 3.72 | 1.67 | ACCOUNT_NUMBER=6, PRODUCT_NAME=2 |
| test | real | 13.46 | 0.00 | none |

## Observations

- Implementation is reproducible and covered by focused unit tests.
- CLI v2 matches simulation output exactly.
- Real slice unchanged, confirming v2 does not alter non-banking outputs in this eval.
- Step 37 + v2 is current best practical pipeline: strong banking WER/entity with lowest observed real regression among tuned LoRA candidates.

## Decision

```
Postprocess v2 implementation accepted.
Current best candidate: Step 37 raw model + Step 42 postprocess v2.

Next (Step 43): re-run v2 postprocess on top model candidates (baseline, Step 33,
Step 37, Step 38) to select best final pipeline by validation/test all + banking
+ real metrics. This isolates model choice from old postprocess v1 effects.
```
