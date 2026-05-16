---
title: "Step 11: Augmented Whisper Evaluation"
type: [feature-note, evaluation]
created: 2026-05-16
status: completed
categories: [asr, evaluation, whisper, mlx, augmentation]
related:
  - step-10-whisper-baseline-postprocessing.md
  - step-9-9router-edge-tts-scaling.md
  - ../../README.md
author: fauzan
---

# Step 11: Augmented Whisper Evaluation

Step ini menjalankan MLX Whisper large-v3 pada 450-row dataset clean+augmented dan membandingkan performa per augmentation profile.

## Spesifikasi

Tujuan minimal:

- Gunakan prediksi clean dari [Step 10](step-10-whisper-baseline-postprocessing.md) sebagai cache.
- Transcribe 300 augmented WAV rows dengan `mlx-community/whisper-large-v3-mlx`.
- Evaluasi raw dan post-processed predictions untuk seluruh 450 rows.
- Pisahkan metrik berdasarkan `dataset_variant` dan `augmentation.name`.

## Dataset

Input manifest:

```text
data/synthetic/9router_edge_dataset_manifest_50.jsonl
```

Dataset composition:

```text
clean: 150
call_low_noise: 150
call_medium_noise: 150
total: 450
```

## Generated Artifacts

Ignored outputs under `data/synthetic/`:

```text
data/synthetic/mlx_whisper_large_v3_predictions_450.jsonl
data/synthetic/mlx_whisper_large_v3_postprocessed_predictions_450.jsonl
data/synthetic/mlx_whisper_large_v3_eval_summary_450.jsonl
data/synthetic/mlx_whisper_large_v3_postprocessed_eval_summary_450.jsonl
data/synthetic/mlx_whisper_large_v3_grouped_eval_450.jsonl
data/synthetic/mlx_whisper_large_v3_postprocessed_grouped_eval_450.jsonl
```

## Overall Results

Raw MLX Whisper large-v3:

```json
{
  "rows": 450,
  "word_errors": 713,
  "reference_words": 4524,
  "wer": 0.15760389036251105,
  "entity_errors": 188,
  "entities": 1395,
  "entity_error_rate": 0.13476702508960572
}
```

Post-processed MLX Whisper large-v3:

```json
{
  "rows": 450,
  "word_errors": 466,
  "reference_words": 4524,
  "wer": 0.10300618921308577,
  "entity_errors": 0,
  "entities": 1395,
  "entity_error_rate": 0.0
}
```

Summary:

```text
Raw WER: 15.76%
Raw entity error rate: 13.48%
Post-processed WER: 10.30%
Post-processed entity error rate: 0.00%
```

## Grouped Results

Raw by dataset slice:

| Slice | Rows | WER | Entity error rate |
|---|---:|---:|---:|
| clean | 150 | 15.72% | 13.12% |
| call_low_noise | 150 | 15.65% | 13.33% |
| call_medium_noise | 150 | 15.92% | 13.98% |

Post-processed by dataset slice:

| Slice | Rows | WER | Entity error rate |
|---|---:|---:|---:|
| clean | 150 | 10.28% | 0.00% |
| call_low_noise | 150 | 10.54% | 0.00% |
| call_medium_noise | 150 | 10.08% | 0.00% |

## Interpretation

The current augmentation profiles barely degrade MLX Whisper large-v3 on this pilot. Raw WER stays around 15.7–15.9% across clean and augmented rows, while the post-processing layer removes all exact entity errors across 1,395 labeled entities.

The result suggests the next bottleneck is not basic noise robustness. It is broader linguistic and entity coverage: more templates, more product names, more merchant names, and more realistic phone-call distortions are needed before fine-tuning conclusions are meaningful.

## Current Limitations

- Augmentation profiles are simple gain/noise transforms, not full telephony simulation.
- Clean and augmented rows share the same TTS source voice, so speaker diversity is still low.
- Post-processing is rule-based and tuned on synthetic banking entities.
- The pilot is only 50 canonical rows expanded to 450 audio rows.

## Next Step

Step 12 should expand dataset diversity before model training: add more banking templates, more voices, and stronger call-channel augmentation. After that, rerun Whisper baseline and decide whether fine-tuning is justified.

## Related

- [Step 10: Whisper Baseline & Banking Post-processing](step-10-whisper-baseline-postprocessing.md)
- [Step 9: 9Router Edge TTS & Generation Scaling](step-9-9router-edge-tts-scaling.md)
- [Project README](../../README.md)
