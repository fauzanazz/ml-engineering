---
title: "Step 48: Keyword Router v1 Spec"
type: [feature-note, artifact, routing]
created: 2026-05-17
status: completed
categories: [asr, whisper, routing, reproducibility]
related:
  - step-47-final-pipeline-comparison-matrix.md
  - step-46-router-trigger-ablation.md
author: fauzan
---

# Step 48: Keyword Router v1 Spec

Step 47 selected `keyword_router_v2` as the best final pipeline candidate. Step
ini turns the chosen router into a reusable artifact/spec so future runs can
recreate the routing rule exactly.

## Artifact

```text
artifacts/step48_keyword_router_v1_spec.json
```

Spec contents:

- `name`: `keyword_router_v1`
- `input`: baseline postprocessed transcript
- `banking_model`: `models/mlx-whisper-large-v3-fullmix-200step-lora-last8-r4-a8-lr1e-4-merged`
- `general_model`: baseline Whisper large-v3 prediction artifacts
- `postprocess`: `banking_entity_v2`
- `regex`: banking keyword regex selected in Step 46
- `final_matrix`: `artifacts/step47_final_pipeline_comparison_v1.json`

## Router Regex

```regex
\b(saldo|rekening|kartu|debit|kredit|pinjaman|kta|paylater|cicilan|qris|bi-fast|bifast|rtgs|transfer|virtual account|bpjs|bunga|interest rate|blokir|terblokir|pin|rupiah|rp\d|juta)\b
```

## Validation

```bash
python3 - <<'PY'
import json, re
spec = json.load(open('artifacts/step48_keyword_router_v1_spec.json'))
re.compile(spec['regex'], re.IGNORECASE)
print(spec['name'], len(spec['keywords']))
PY
```

Result:

```text
keyword_router_v1 24
```

## Reproducibility

Final routed prediction assembly requires four input prediction sets:

1. Baseline raw predictions: `artifacts/mlx_whisper_large_v3_baseline_combined_10h_{split}_predictions.jsonl`
2. Step37 raw predictions: `artifacts/mlx_whisper_large_v3_fullmix_200step_lora_last8_r4_a8_lr1e-4_merged_{split}_predictions.jsonl`
3. Postprocess v2 code: `src/indonesian_banking_asr/evaluation/postprocess.py`
4. Router spec: `artifacts/step48_keyword_router_v1_spec.json`

Pipeline logic:

```text
baseline_raw -> postprocess_v2 -> keyword_router_v1
if router matches:
    output Step37 raw -> postprocess_v2
else:
    output baseline raw -> postprocess_v2
```

## Decision

```
Router spec accepted. The final candidate is now reproducible from artifacts and
source code.

Next (Step 49): write final model-selection summary with recommended production
candidate, known limitations, and next follow-up experiments.
```
