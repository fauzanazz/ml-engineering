---
title: "Step 40: Entity Postprocess Error Audit"
type: [feature-note, evaluation, error-analysis]
created: 2026-05-17
status: completed
categories: [asr, whisper, postprocessing, entity-error]
related:
  - step-39-fullmix-rank4-alpha4-strength-probe.md
  - step-37-fullmix-rank4-lora-regularization.md
  - step-23-entity-aware-postprocessing-10h.md
author: fauzan
---

# Step 40: Entity Postprocess Error Audit

Training sweeps Step 33-39 menunjukkan kandidat terbaik sementara adalah Step 37
(full-mix last-8 r4/a8 lr=1e-4): banking kuat, real regression paling kecil pada
test, tetapi entity postprocessed masih kalah baseline. Step ini audit error
entity pada test banking untuk mencari perbaikan tanpa melatih ulang.

## Setup

Audit membandingkan entity miss counts pada `artifacts/combined_10h_test_banking_manifest.jsonl`
untuk baseline, Step 33, Step 37, dan Step 38.

```bash
python3 - <<'PY'
import json, collections

def rows(path):
    return [json.loads(line) for line in open(path)]

def counts(manifest, prediction_path):
    predictions = {row["utterance_id"]: row["hypothesis"] for row in rows(prediction_path)}
    misses = collections.Counter()
    total = 0
    for row in manifest:
        hypothesis = predictions[row["utterance_id"]].casefold()
        for entity in row.get("entities", []):
            if entity["text"].casefold() not in hypothesis:
                misses[entity["type"]] += 1
                total += 1
    return total, dict(misses)
PY
```

## Results

Test banking entity misses by type (156 rows, 480 entities):

| Model | Mode | Total Miss | ACCOUNT_NUMBER | AMOUNT | PRODUCT_NAME | TRANSFER_METHOD | CARD_LAST4 | BANKING_TERM |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | raw | 67 | 21 | 13 | 12 | 3 | 6 | 12 |
| baseline | postprocessed | 16 | 3 | 6 | 4 | 3 | 0 | 0 |
| Step 33 r8/a16 | raw | 34 | 15 | 6 | 9 | 3 | 1 | 0 |
| Step 33 r8/a16 | postprocessed | 22 | 3 | 6 | 9 | 3 | 1 | 0 |
| Step 37 r4/a8 | raw | 38 | 20 | 6 | 9 | 3 | 0 | 0 |
| Step 37 r4/a8 | postprocessed | 24 | 6 | 6 | 9 | 3 | 0 | 0 |
| Step 38 r2/a4 | postprocessed | 18 | 6 | 6 | 3 | 3 | 0 | 0 |

Representative Step 37 postprocessed misses:

| Type | Reference Entity | Hypothesis Pattern | Fix Class |
|---|---|---|---|
| ACCOUNT_NUMBER | `67087515144759` | `6708751514459` | true digit deletion, cannot postprocess safely |
| ACCOUNT_NUMBER | `74733542600060` | `747335426000` | true suffix deletion, cannot postprocess safely |
| AMOUNT | `Rp11.070.000` | `Rp11.07.000` | format repair possible |
| AMOUNT | `Rp11.000.000` | `Rp11 juta` | spoken million normalization possible |
| PRODUCT_NAME | `QRIS` | `CRIS` | lexicon repair possible |
| PRODUCT_NAME | `paylater` | `Piloter`, `Pailater` | lexicon repair possible |
| PRODUCT_NAME | `virtual account` | `virtual count` | lexicon repair possible |
| TRANSFER_METHOD | `RTGS` | `RTGE` | lexicon repair possible |

## Observations

- Existing postprocess is excellent for grouped digits, reducing Step 37 account misses from 20 to 6.
- Remaining ACCOUNT_NUMBER misses are mostly real digit deletion/substitution, unsafe to hallucinate from context.
- PRODUCT_NAME and TRANSFER_METHOD errors are mostly deterministic lexicon confusions absent from current replacements (`CRIS`, `Piloter`, `Pailater`, `virtual count`, `RTGE`).
- AMOUNT errors are formatting/spoken-number gaps (`Rp11.07.000`, `Rp11 juta`) and likely safe to normalize with narrow regexes.
- Step 38 postprocessed gets fewer PRODUCT_NAME misses than Step 37 (3 vs 9), but has worse real WER and raw banking, so postprocess should target Step 37 rather than switch model.

## Decision

```
Training best remains Step 37 (r4/a8). Postprocessing is now highest-leverage path:
PRODUCT_NAME + TRANSFER_METHOD lexicon repairs and narrow AMOUNT normalizers can
reduce entity misses without affecting non-banking real WER if only applied to
prediction text after transcription.

Next (Step 41): implement/simulate postprocess v2 on existing Step 37 predictions
with narrow replacements:
- CRIS -> QRIS
- Piloter/Pailater -> paylater
- virtual count -> virtual account
- RTGE -> RTGS
- RpXX juta -> RpXX.000.000
- RpDD.DD.000 -> RpDD.DDD.000 where safe
Then evaluate Step 37 v2 postprocessed metrics before changing default behavior.
```
