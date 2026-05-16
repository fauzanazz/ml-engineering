---
title: "Step 13: Combined Training Manifest"
type: [feature-note, training-data]
created: 2026-05-16
status: completed
categories: [asr, training, manifest, babelspeech, synthetic-banking]
related:
  - step-12-babelspeech-real-data-grounding.md
  - step-11-augmented-whisper-evaluation.md
  - ../../README.md
author: fauzan
---

# Step 13: Combined Training Manifest

Step ini membuat combined training manifest dari BabelSpeech real colloquial rows dan synthetic banking rows. Ini belum fine-tuning model; outputnya adalah manifest training deterministic yang siap dipakai fine-tuning Whisper/ASR trainer berikutnya.

## Goal

Use Step 12 mix plan:

```text
70% BabelSpeech general Indonesian
20% synthetic banking clean
10% synthetic banking augmented
```

Because current BabelSpeech artifact is a 50-row sample with 40 train rows, the mixer scales to available train rows and keeps only `split == "train"`.

## Implementation

Added:

```text
src/indonesian_banking_asr/training/combined.py
tests/training/test_combined.py
```

CLI:

```text
banking-asr-build-combined-training
```

Generated ignored artifacts:

```text
data/training/combined_train_manifest_57.jsonl
data/training/combined_train_summary_57.jsonl
```

## Command

```bash
uv run banking-asr-build-combined-training \
  --babelspeech-manifest-path data/real/babelspeech_manifest_50.jsonl \
  --synthetic-manifest-path data/synthetic/9router_edge_dataset_manifest_50.jsonl \
  --output-path data/training/combined_train_manifest_57.jsonl \
  --summary-path data/training/combined_train_summary_57.jsonl
```

## Result

```json
{
  "total_rows": 57,
  "training_source_counts": {
    "babelspeech_general": 40,
    "synthetic_banking_clean": 11,
    "synthetic_banking_augmented": 6
  },
  "dataset_variant_counts": {
    "real": 40,
    "clean": 11,
    "augmented": 6
  },
  "split_counts": {
    "train": 57
  }
}
```

Approximate mix:

```text
BabelSpeech: 70.18%
Synthetic clean: 19.30%
Synthetic augmented: 10.53%
```

## WER Status

No new combined-data WER exists yet because no model has been fine-tuned on this combined manifest. Existing eval baselines remain:

```text
BabelSpeech sample raw WER: 11.93%
Synthetic banking clean+aug raw WER: 15.76%
Synthetic banking clean+aug post-processed WER: 10.30%
```

The next measurable WER requires a fine-tuning step that consumes `data/training/combined_train_manifest_57.jsonl`, then evaluates on held-out BabelSpeech and synthetic banking test sets.

## Tests

```bash
uv run pytest tests/training/test_combined.py
```

Result:

```text
2 passed
```
