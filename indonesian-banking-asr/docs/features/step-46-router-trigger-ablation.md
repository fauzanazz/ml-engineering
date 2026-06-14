---
title: "Step 46: Router Trigger Ablation"
type: [feature-note, evaluation, routing, ablation]
created: 2026-05-17
status: completed
categories: [asr, whisper, routing, model-selection]
related:
  - step-45-heuristic-domain-router.md
  - step-44-oracle-banking-router-upper-bound.md
author: fauzan
---

# Step 46: Router Trigger Ablation

Step 45 showed heuristic routing beats the oracle upper bound on all-WER in this
split. Step ini ablates trigger sets to verify the rule is stable and not relying
on brittle long-digit false positives.

## Variants

- `keyword_only`: full Step 45 banking/domain keyword list, no standalone long-digit fallback.
- `digit_only`: `\d{8,}|Rp\d|juta` only.
- `strict_keyword`: smaller high-precision keyword list.
- `full`: Step 45 rule (`keyword_only OR \d{8,}`).

All variants route baseline+v2 transcript -> Step37+v2 if matched, else
baseline+v2.

## Commands

```bash
python3 - <<'PY'
# Build routed predictions for keyword_only, digit_only, strict_keyword, full.
PY

for variant in keyword_only digit_only strict_keyword full; do
  P=artifacts/router_ablation_${variant}
  for split in validation test; do
    for slice in "" "_banking" "_real"; do
      uv run banking-asr-evaluate \
        --manifest-path artifacts/combined_10h_${split}${slice}_manifest.jsonl \
        --predictions-path ${P}_${split}_postprocessed_predictions.jsonl \
        --output-path ${P}_${split}${slice}_postprocessed_eval.jsonl
    done
  done
done
```

## Routing Stats

| Variant | Split | True Banking Routed | True Banking Missed | Real False Positives |
|---|---|---:|---:|---:|
| keyword_only | val | 120 | 12 | 10 |
| keyword_only | test | 153 | 3 | 12 |
| digit_only | val | 69 | 63 | 6 |
| digit_only | test | 108 | 48 | 9 |
| strict_keyword | val | 117 | 15 | 3 |
| strict_keyword | test | 132 | 24 | 2 |
| full | val | 120 | 12 | 10 |
| full | test | 153 | 3 | 12 |

## Results

v2 postprocessed WER / entity-error (%):

| Split | Variant | All | Banking | Real |
|---|---|---|---|---|
| val | keyword_only | 11.11 / 2.88 | 4.19 / 2.88 | 12.25 / 0.00 |
| val | digit_only | 11.21 / 0.72 | 3.53 / 0.72 | 12.48 / 0.00 |
| val | strict_keyword | 11.14 / 2.88 | 4.19 / 2.88 | 12.29 / 0.00 |
| val | full | 11.11 / 2.88 | 4.19 / 2.88 | 12.25 / 0.00 |
| test | keyword_only | 11.35 / 1.67 | 3.72 / 1.67 | 12.72 / 0.00 |
| test | digit_only | 11.54 / 1.67 | 5.08 / 1.67 | 12.70 / 0.00 |
| test | strict_keyword | 11.68 / 1.67 | 5.29 / 1.67 | 12.83 / 0.00 |
| test | full | 11.35 / 1.67 | 3.72 / 1.67 | 12.72 / 0.00 |

## Observations

- `keyword_only` and `full` are identical on both splits; long-digit fallback adds no routing beyond the keyword list in this dataset.
- `digit_only` looks good on validation banking but misses too many test banking rows (48/156), causing test banking WER 5.08 vs 3.72.
- `strict_keyword` reduces false positives but hurts test banking recall and all-WER.
- Best stable rule is `keyword_only`: same metrics as full, simpler, less brittle, no standalone digit trigger.

## Decision

```
Use keyword-only router. Drop standalone long-digit fallback.
Chosen practical pipeline:
baseline+v2 transcript -> keyword-only router -> Step37+v2 for banking-like text,
otherwise baseline+v2.

Next (Step 47): package final pipeline artifacts and produce final comparison
matrix vs baseline, Step37 single-model, oracle route, and keyword router.
```
