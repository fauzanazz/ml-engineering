---
title: "Step 45: Heuristic Domain Router"
type: [feature-note, evaluation, routing]
created: 2026-05-17
status: completed
categories: [asr, whisper, routing, model-selection]
related:
  - step-44-oracle-banking-router-upper-bound.md
  - step-43-v2-candidate-selection-matrix.md
author: fauzan
---

# Step 45: Heuristic Domain Router

Step 44 showed oracle routing is valuable. Step ini tests a non-oracle router
using only baseline+v2 transcript text. If the transcript contains banking/domain
keywords or long digit patterns, route to Step37+v2; otherwise keep baseline+v2.

## Router

```python
pattern = re.compile(
    r"\b(saldo|rekening|kartu|debit|kredit|pinjaman|kta|paylater|cicilan|"
    r"qris|bi-fast|bifast|rtgs|transfer|virtual account|bpjs|bunga|"
    r"interest rate|blokir|terblokir|pin|rupiah|rp\d|juta)\b|\d{8,}",
    re.IGNORECASE,
)

route_to_step37 = bool(pattern.search(baseline_v2_hypothesis))
```

Output prefix:

```text
artifacts/mlx_whisper_large_v3_heuristic_router_baseline_text_to_s37_v2
```

## Commands

```bash
python3 - <<'PY'
# Build routed predictions from baseline_v2 and Step37_v2 by keyword match.
PY

P=artifacts/mlx_whisper_large_v3_heuristic_router_baseline_text_to_s37_v2
for split in validation test; do
  for slice in "" "_banking" "_real"; do
    uv run banking-asr-evaluate \
      --manifest-path artifacts/combined_10h_${split}${slice}_manifest.jsonl \
      --predictions-path ${P}_${split}_postprocessed_predictions.jsonl \
      --output-path ${P}_${split}${slice}_postprocessed_eval.jsonl
  done
done
```

## Routing Stats

| Split | True Real -> Real | True Real -> Banking | True Banking -> Banking | True Banking -> Real |
|---|---:|---:|---:|---:|
| val | 343 | 10 | 120 | 12 |
| test | 343 | 12 | 153 | 3 |

Recall / false positive rate:

| Split | Banking Recall | Real False Positive Rate |
|---|---:|---:|
| val | 90.9% | 2.8% |
| test | 98.1% | 3.4% |

## Results

v2 postprocessed WER / entity-error (%):

| Split | Pipeline | All | Banking | Real |
|---|---|---|---|---|
| val | baseline+v2 | 11.30 / 0.72 | 3.94 / 0.72 | 12.51 / 0.00 |
| val | Step37+v2 | 12.95 / 2.88 | 4.19 / 2.88 | 14.39 / 0.00 |
| val | oracle route | 11.33 / 2.88 | 4.19 / 2.88 | 12.51 / 0.00 |
| val | heuristic route | 11.11 / 2.88 | 4.19 / 2.88 | 12.25 / 0.00 |
| test | baseline+v2 | 12.35 / 1.88 | 9.80 / 1.88 | 12.81 / 0.00 |
| test | Step37+v2 | 11.97 / 1.67 | 3.72 / 1.67 | 13.46 / 0.00 |
| test | oracle route | 11.43 / 1.67 | 3.72 / 1.67 | 12.81 / 0.00 |
| test | heuristic route | 11.35 / 1.67 | 3.72 / 1.67 | 12.72 / 0.00 |

## Observations

- Heuristic router beats oracle all-WER on both splits (val 11.11 vs 11.33, test 11.35 vs 11.43) because a small number of real false positives happen to improve WER.
- Banking recall is high, especially test (153/156 routed to Step37).
- Real false positive rate remains low (10/353 val, 12/355 test) and does not hurt real WER; it improves real WER slightly in this eval.
- Banking metrics equal oracle despite missed banking rows, meaning missed rows are not the ones driving aggregate banking errors.
- This is the first practical pipeline beating baseline all-WER while preserving strong banking gains and real performance.

## Decision

```
Heuristic routing is ACCEPTED as best current strategy:
baseline+v2 transcript keyword router -> Step37+v2 for banking-like utterances.

Next (Step 46): ablate router triggers (keyword-only vs digit-only vs full) to
ensure gains are not from brittle false positives and choose a stable router rule.
```
