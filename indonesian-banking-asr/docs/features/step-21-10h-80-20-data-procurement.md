---
title: "Step 21: 10h 80/20 Data Procurement"
type: [feature-note, data, procurement]
created: 2026-05-16
status: completed
categories: [asr, dataset, babelspeech, synthetic-data, tts]
related:
  - step-20-guarded-sgd-scaleup-evaluation.md
  - step-13-combined-training-manifest.md
  - ../../README.md
author: fauzan
---

# Step 21: 10h 80/20 Data Procurement

Step ini memperbesar dataset dari kurang dari 1 jam menjadi kandidat 10 jam dengan target komposisi 80% Indonesian non-banking dan 20% synthetic banking.

## Rationale

Whisper large-v3 sudah kuat untuk Indonesian general speech. Non-banking data tetap dipakai sebagai regularizer agar fine-tuning banking tidak overfit ke synthetic banking style dan tidak merusak general Indonesian recognition.

## Non-banking Real Data

Source:

```text
BabelSpeech/40hours_Indonesian_Colloquial_ASR_Speech_Dataset
```

Procured 8-hour slice from local Hugging Face cache:

```bash
uv run banking-asr-convert-babelspeech \
  --dataset-root /Users/fauzan/.cache/huggingface/hub/datasets--BabelSpeech--40hours_Indonesian_Colloquial_ASR_Speech_Dataset/snapshots/7af54dbe9e40a888bc9fc92a183308e7ee78f14a \
  --output-path data/real/babelspeech_manifest_8h.jsonl \
  --extract-audio \
  --limit 3539
```

Result:

| Split | Rows | Hours |
|---|---:|---:|
| train | 2,831 | 6.4269 |
| validation | 353 | 0.7675 |
| test | 355 | 0.8059 |
| total | 3,539 | 8.0003 |

## Banking Synthetic Data

9Router paraphrase endpoint returned 404 during this run, so banking expansion uses deterministic dry-run text generation plus Edge TTS. This is acceptable for procurement, but lower linguistic diversity than live Gemini/9Router paraphrases.

Commands:

```bash
uv run banking-asr-generate-text \
  --output-path data/synthetic/text_manifest_432_dryrun.jsonl \
  --accepted-output-path data/synthetic/accepted_manifest_432_dryrun.jsonl \
  --rejected-output-path data/synthetic/rejected_manifest_432_dryrun.jsonl \
  --summary-output-path data/synthetic/summary_manifest_432_dryrun.jsonl \
  --paraphrase-mode dry-run \
  --samples-per-template 36 \
  --variant-count 3
```

```bash
uv run banking-asr-generate-text tts \
  --input-path data/synthetic/accepted_manifest_432_dryrun.jsonl \
  --output-path data/synthetic/edge_audio_manifest_432_dryrun.jsonl \
  --audio-dir data/synthetic/edge_audio_clean_432_dryrun \
  --provider edge \
  --voice id-ID-ArdiNeural \
  --sample-rate 16000 \
  --resume \
  --seconds-per-request 0.05
```

```bash
uv run banking-asr-generate-text augment-audio \
  --input-path data/synthetic/edge_audio_manifest_432_dryrun.jsonl \
  --output-path data/synthetic/edge_augmented_manifest_432_dryrun.jsonl \
  --output-dir data/synthetic/edge_audio_augmented_432_dryrun \
  --profile call_low_noise:0.85:120:42 \
  --profile call_medium_noise:0.75:260:43
```

Result:

| Split | Rows | Hours |
|---|---:|---:|
| train | 1,008 | 1.6192 |
| validation | 132 | 0.2225 |
| test | 156 | 0.2595 |
| total | 1,296 | 2.1012 |

QA:

```json
{"checked_rows": 1296, "valid_rows": 1296, "invalid_rows": 0, "errors": []}
```

## Combined Candidate

Full candidate manifest:

```text
data/training/combined_manifest_10h_80_20_candidate.jsonl
```

Train-only shuffled candidate manifest:

```text
data/training/combined_train_manifest_10h_80_20_candidate.jsonl
```

Full candidate totals:

| Source | Rows | Hours | Share |
|---|---:|---:|---:|
| non-banking real | 3,539 | 8.0003 | 79.20% |
| banking synthetic | 1,296 | 2.1012 | 20.80% |
| total | 4,835 | 10.1015 | 100.00% |

Train-only candidate totals:

| Source | Rows | Hours | Share |
|---|---:|---:|---:|
| non-banking real | 2,831 | 6.4269 | 79.88% |
| banking synthetic | 1,008 | 1.6192 | 20.12% |
| total | 3,839 | 8.0462 | 100.00% |

## Decision

Use `combined_train_manifest_10h_80_20_candidate.jsonl` for next training experiments, but treat dry-run banking text as a temporary procurement fill. Before serious model comparison, repair 9Router/Gemini paraphrase generation or add another live paraphrase provider for stronger banking variation.

Recommended Step 22:

```text
run baseline eval on 10h candidate validation/test slices, then start adapter/LoRA-style training instead of direct decoder-only updates
```
