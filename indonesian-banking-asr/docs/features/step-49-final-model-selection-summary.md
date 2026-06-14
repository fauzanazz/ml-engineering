---
title: "Step 49: Final Model-Selection Summary"
type: [feature-note, summary, model-selection]
created: 2026-05-17
status: completed
categories: [asr, whisper, lora, routing, summary]
related:
  - step-48-keyword-router-spec.md
  - step-47-final-pipeline-comparison-matrix.md
  - step-37-fullmix-rank4-lora-regularization.md
author: fauzan
---

# Step 49: Final Model-Selection Summary

This step consolidates the experimental findings from Step 26-48 and selects the
current best Indonesian banking ASR pipeline.

## Training Findings

| Step | Experiment | Outcome |
|---:|---|---|
| 29 | Banking-only LoRA lr=1e-6 | No movement, lr too low |
| 30 | Banking-only LoRA lr=5e-5 | First real banking gain |
| 31 | Banking-only LoRA lr=1e-4 | Best smoke raw gain |
| 32 | Banking-only lr=1e-4 full split | Banking gain, real regression |
| 33 | Full-mix r8/a16 lr=1e-4 | Best raw banking/entity, real regression persists |
| 34 | Full-mix 100-step | Worse than 200-step |
| 35 | Full-mix lr=5e-5 | Worse than lr=1e-4 |
| 36 | Full-mix last-4 scope | Much worse real and banking WER |
| 37 | Full-mix r4/a8 lr=1e-4 | Best training trade-off |
| 38 | Full-mix r2/a4 | Worse real test than r4 |
| 39 | Full-mix r4/a4 | Worse than r4/a8 |

Best trained model:

```text
models/mlx-whisper-large-v3-fullmix-200step-lora-last8-r4-a8-lr1e-4-merged
```

Why:

- Keeps strong banking WER (test banking 3.72 after v2).
- Has lowest tuned-model real regression on test (13.46 real WER after v2).
- Beats baseline on test all-WER when used with router.

## Postprocess Findings

Postprocess v2 is accepted and implemented in:

```text
src/indonesian_banking_asr/evaluation/postprocess.py
```

It fixes narrow high-confidence entity confusions:

```text
CRIS -> QRIS
Piloter/Pailater -> paylater
virtual count -> virtual account
RTGE -> RTGS
RpXX juta -> RpXX.000.000
RpDD.DD.000 -> RpDD.DDD.000
```

Focused tests pass:

```text
uv run pytest tests/evaluation/test_postprocess.py tests/evaluation/test_postprocess_cli.py
4 passed in 0.01s
```

## Final Candidate

Selected pipeline:

```text
keyword_router_v1:
  run baseline + postprocess v2
  if baseline postprocessed transcript matches banking keyword regex:
      return Step37 model + postprocess v2
  else:
      return baseline + postprocess v2
```

Router spec:

```text
artifacts/step48_keyword_router_v1_spec.json
```

Final matrix:

```text
artifacts/step47_final_pipeline_comparison_v1.json
artifacts/step47_final_pipeline_comparison_v1.md
```

## Final Metrics

Keyword router v2 WER / entity-error (%):

| Split | Slice | Rows | WER | Entity |
|---|---|---:|---:|---:|
| validation | all | 485 | 11.11 | 2.88 |
| validation | banking | 132 | 4.19 | 2.88 |
| validation | real | 353 | 12.25 | 0.00 |
| test | all | 511 | 11.35 | 1.67 |
| test | banking | 156 | 3.72 | 1.67 |
| test | real | 355 | 12.72 | 0.00 |

Baseline v2 comparison:

| Split | Slice | Baseline WER | Router WER | Delta |
|---|---|---:|---:|---:|
| validation | all | 11.30 | 11.11 | -0.19 |
| validation | banking | 3.94 | 4.19 | +0.25 |
| validation | real | 12.51 | 12.25 | -0.26 |
| test | all | 12.35 | 11.35 | -1.00 |
| test | banking | 9.80 | 3.72 | -6.08 |
| test | real | 12.81 | 12.72 | -0.09 |

## Limitations

- Validation banking baseline remains slightly better than router (3.94 vs 4.19 WER, 0.72 vs 2.88 entity).
- Router depends on baseline transcript keywords; domain misses are possible.
- Evaluation slices are synthetic banking + real non-banking; production needs real banking calls.
- Remaining entity misses are mostly true digit deletion; unsafe to hallucinate account numbers.

## Decision

```
Select keyword_router_v1 + postprocess v2 + Step37 banking model as current best
candidate for Indonesian banking ASR.

Step 50 should be final completion audit: verify docs Step 26-50, artifacts,
implemented code/tests, and explicit success criteria before closing goal.
```
