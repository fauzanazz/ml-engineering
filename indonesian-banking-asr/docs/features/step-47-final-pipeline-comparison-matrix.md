---
title: "Step 47: Final Pipeline Comparison Matrix v1"
type: [feature-note, evaluation, model-selection, artifact]
created: 2026-05-17
status: completed
categories: [asr, whisper, routing, final-matrix]
related:
  - step-46-router-trigger-ablation.md
  - step-45-heuristic-domain-router.md
  - step-42-postprocess-v2-implementation.md
author: fauzan
---

# Step 47: Final Pipeline Comparison Matrix v1

Step 46 selected the keyword-only router. Step ini packages the final comparison
matrix across the practical options: baseline+v2, Step37 single-model+v2, oracle
route, and keyword router.

## Artifacts

- `artifacts/step47_final_pipeline_comparison_v1.json`
- `artifacts/step47_final_pipeline_comparison_v1.md`

## Commands

```bash
python3 - <<'PY'
# Collect postprocessed eval JSONL metrics into JSON + Markdown matrix.
PY
```

## Results

| Split | Pipeline | Slice | Rows | WER % | Entity % |
|---|---|---|---:|---:|---:|
| validation | baseline_v2 | all | 485 | 11.30 | 0.72 |
| validation | baseline_v2 | banking | 132 | 3.94 | 0.72 |
| validation | baseline_v2 | real | 353 | 12.51 | 0.00 |
| validation | step37_single_v2 | all | 485 | 12.95 | 2.88 |
| validation | step37_single_v2 | banking | 132 | 4.19 | 2.88 |
| validation | step37_single_v2 | real | 353 | 14.39 | 0.00 |
| validation | oracle_route_v2 | all | 485 | 11.33 | 2.88 |
| validation | oracle_route_v2 | banking | 132 | 4.19 | 2.88 |
| validation | oracle_route_v2 | real | 353 | 12.51 | 0.00 |
| validation | keyword_router_v2 | all | 485 | 11.11 | 2.88 |
| validation | keyword_router_v2 | banking | 132 | 4.19 | 2.88 |
| validation | keyword_router_v2 | real | 353 | 12.25 | 0.00 |
| test | baseline_v2 | all | 511 | 12.35 | 1.88 |
| test | baseline_v2 | banking | 156 | 9.80 | 1.88 |
| test | baseline_v2 | real | 355 | 12.81 | 0.00 |
| test | step37_single_v2 | all | 511 | 11.97 | 1.67 |
| test | step37_single_v2 | banking | 156 | 3.72 | 1.67 |
| test | step37_single_v2 | real | 355 | 13.46 | 0.00 |
| test | oracle_route_v2 | all | 511 | 11.43 | 1.67 |
| test | oracle_route_v2 | banking | 156 | 3.72 | 1.67 |
| test | oracle_route_v2 | real | 355 | 12.81 | 0.00 |
| test | keyword_router_v2 | all | 511 | 11.35 | 1.67 |
| test | keyword_router_v2 | banking | 156 | 3.72 | 1.67 |
| test | keyword_router_v2 | real | 355 | 12.72 | 0.00 |

## Observations

- Keyword router is best all-WER on both validation and test: 11.11 validation, 11.35 test.
- Keyword router preserves banking gains from Step37: test banking WER 3.72 vs baseline 9.80.
- Keyword router preserves/improves real WER vs baseline in this eval: validation 12.25 vs 12.51, test 12.72 vs 12.81.
- Entity error for keyword router follows Step37 on routed banking rows: 2.88 validation, 1.67 test.
- Baseline still has best validation banking entity (0.72), but much worse test banking WER.

## Decision

```
Select keyword_router_v2 as current best final pipeline candidate.
Final candidate definition:
1. Run baseline Whisper large-v3 + postprocess v2.
2. If baseline postprocessed transcript matches banking keyword router, use Step37
   model output + postprocess v2 instead.
3. Otherwise keep baseline output + postprocess v2.

Next (Step 48): convert keyword router into a reusable local artifact/spec and
add an explicit reproducibility note for final pipeline assembly.
```
