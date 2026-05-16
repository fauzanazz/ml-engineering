---
title: "Step 23: Entity-aware Post-processing on 10h Baseline"
type: [feature-note, evaluation, postprocessing]
created: 2026-05-16
status: completed
categories: [asr, whisper, evaluation, postprocessing, banking-entities]
related:
  - step-22-10h-baseline-evaluation.md
  - step-10-whisper-baseline-postprocessing.md
  - ../../README.md
author: fauzan
---

# Step 23: Entity-aware Post-processing on 10h Baseline

Step ini menerapkan banking entity post-processing ke prediksi baseline MLX Whisper large-v3 dari Step 22. Tujuannya mengukur seberapa jauh rule-based post-processing bisa menutup gap entity sebelum training adapter/LoRA.

## Implementation

Added repeatable CLI:

```text
banking-asr-postprocess-predictions
```

Files:

```text
src/indonesian_banking_asr/evaluation/postprocess_cli.py
tests/evaluation/test_postprocess_cli.py
```

The CLI preserves raw output in `raw_hypothesis`, writes post-processed text to `hypothesis`, and tags rows with `postprocess=banking_entity_v1`.

## Commands

```bash
uv run banking-asr-postprocess-predictions \
  --predictions-path artifacts/mlx_whisper_large_v3_baseline_combined_10h_validation_predictions.jsonl \
  --output-path artifacts/mlx_whisper_large_v3_baseline_combined_10h_validation_postprocessed_predictions.jsonl
```

```bash
uv run banking-asr-postprocess-predictions \
  --predictions-path artifacts/mlx_whisper_large_v3_baseline_combined_10h_test_predictions.jsonl \
  --output-path artifacts/mlx_whisper_large_v3_baseline_combined_10h_test_postprocessed_predictions.jsonl
```

Changed rows:

```text
validation: 58 / 485
test: 56 / 511
```

## Results

Combined metrics:

| Split | Metric | Raw baseline | Post-processed | Relative change |
|---|---|---:|---:|---:|
| validation | WER | 12.11% | 11.33% | -6.4% |
| validation | entity error rate | 17.27% | 1.44% | -91.7% |
| test | WER | 13.02% | 12.43% | -4.5% |
| test | entity error rate | 13.96% | 3.33% | -76.1% |

Banking-only metrics:

| Split | Metric | Raw baseline | Post-processed | Relative change |
|---|---|---:|---:|---:|
| validation banking | WER | 9.69% | 4.19% | -56.8% |
| validation banking | entity error rate | 17.27% | 1.44% | -91.7% |
| test banking | WER | 14.23% | 10.30% | -27.6% |
| test banking | entity error rate | 13.96% | 3.33% | -76.1% |

Real non-banking metrics stayed effectively unchanged:

| Split | Raw WER | Post-processed WER |
|---|---:|---:|
| validation real | 12.51% | 12.51% |
| test real | 12.80% | 12.81% |

Remaining validation banking entity errors:

```json
{"ACCOUNT_NUMBER": 3, "TRANSFER_METHOD": 3}
```

Remaining test banking entity errors:

```json
{"AMOUNT": 6, "PRODUCT_NAME": 4, "TRANSFER_METHOD": 3, "ACCOUNT_NUMBER": 3}
```

## Interpretation

Post-processing gives the first clear improvement over baseline on the 10h candidate. It sharply reduces banking entity errors while leaving real non-banking WER almost unchanged.

This confirms the project target should separate two layers:

```text
ASR model: keep Indonesian robustness
banking normalization layer: enforce critical entity fidelity
```

## Decision

Keep post-processing as the default evaluation layer for banking metrics. Next training experiments should compare against both raw Whisper and post-processed Whisper, because beating the post-processed baseline is much harder and more relevant.

Recommended Step 24:

```text
adapter/LoRA training on 10h train manifest, evaluated against raw and post-processed baselines
```
