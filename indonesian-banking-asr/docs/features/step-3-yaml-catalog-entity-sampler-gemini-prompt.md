---
title: "Step 3: YAML Catalog, Entity Sampler & Gemini Prompt Test"
type: [feature-note, implementation]
created: 2026-05-13
status: completed
categories: [asr, dataset, yaml, synthetic-data, gemini, deterministic-split]
related:
  - step-2-synthetic-text-generator-and-validator.md
  - ../../README.md
author: fauzan
---

# Step 3: YAML Catalog, Entity Sampler & Gemini Prompt Test

Step ini memperluas generator synthetic banking ASR dari satu pilot row menjadi catalog-driven generation: YAML template catalog, deterministic entity sampler, deterministic split assignment, multi-row manifest pipeline, dan Gemini prompt/response parser test.

## Spesifikasi

Keputusan dari planning:

- Template format: YAML.
- Dependency: `PyYAML` via `uv add PyYAML`.
- Scope: kecil dulu untuk validasi.
- Product values: generic, bukan nama bank/product spesifik.
- Split: deterministic.
- Gemini: test prompt/parse dulu, belum call API live.

## Files Added

```text
indonesian-banking-asr/
├── data/
│   └── templates/
│       └── banking_intents.yaml
├── src/
│   └── indonesian_banking_asr/
│       └── synthetic/
│           ├── catalog.py
│           ├── gemini.py
│           ├── pipeline.py
│           ├── sampler.py
│           └── split.py
└── tests/
    └── synthetic/
        ├── test_catalog.py
        ├── test_gemini.py
        ├── test_pipeline.py
        ├── test_sampler.py
        └── test_split.py
```

Updated:

```text
pyproject.toml
uv.lock
src/indonesian_banking_asr/synthetic/cli.py
tests/synthetic/test_cli.py
README.md
```

## YAML Template Catalog

Default catalog:

```text
data/templates/banking_intents.yaml
```

Initial coverage: 12 intents, one template each.

| Intent | Template ID |
|---|---|
| `check_balance` | `check_balance_001` |
| `check_transaction_history` | `check_transaction_history_001` |
| `check_installment` | `check_installment_001` |
| `credit_card_limit` | `credit_card_limit_001` |
| `loan_interest_rate` | `loan_interest_rate_001` |
| `mortgage_interest_rate` | `mortgage_interest_rate_001` |
| `account_blocked` | `account_blocked_001` |
| `card_blocked` | `card_blocked_001` |
| `transfer_failed` | `transfer_failed_001` |
| `virtual_account_payment` | `virtual_account_payment_001` |
| `change_phone_number` | `change_phone_number_001` |
| `complaint_fee_charge` | `complaint_fee_charge_001` |

Example YAML row:

```yaml
- template_id: check_installment_001
  intent: check_installment
  text: "Saya mau cek cicilan {product_name} sebesar {amount}."
  entities:
    - type: BANKING_TERM
      slot: cicilan
      value: cicilan
    - type: PRODUCT_NAME
      slot: product_name
    - type: AMOUNT
      slot: amount
```

## Entity Sampler

`EntitySampler(seed=...)` generates deterministic values for slots:

```text
product_name
account_number
amount
interest_rate
date
card_last4
tenor
transfer_method
merchant_name
```

Generic product list:

```text
rekening tabungan
rekening giro
deposito
kartu kredit
kartu debit
KPR
KTA
pinjaman multiguna
cicilan kendaraan
virtual account
BI-FAST
mobile banking
internet banking
QRIS
paylater
```

Entity ranges:

```text
ACCOUNT_NUMBER: 10-14 digit
AMOUNT: Rp10.000 - Rp50.000.000
INTEREST_RATE: 2,5% - 18,0%
CARD_LAST4: 4 digit
TENOR: 3, 6, 12, 24, 36, 60 bulan
```

## Deterministic Split

Split assignment:

```text
sha256(template_id|account_number|amount) -> stable bucket
```

Thresholds:

```text
train: bucket < 0.8
validation: 0.8 <= bucket < 0.9
test: bucket >= 0.9
```

Why:

- Reproducible.
- Less random leakage.
- Split depends on template + key entities, not row order.

## Gemini Prompt Test

Step 3 adds Gemini helper only:

```text
build_paraphrase_prompt()
parse_gemini_json_array()
```

No live Gemini API call yet.

Prompt still enforces:

```text
Keep exact entity strings unchanged.
Do not change numbers.
Do not add new account numbers.
Do not add new money amounts.
Output JSON array only.
```

Parser accepts only JSON array of strings.

## CLI

Generate manifest rows:

```bash
uv run banking-asr-generate-text \
  --output-path data/synthetic/manifests/pilot.jsonl \
  --seed 123 \
  --limit 12
```

Module form:

```bash
uv run python -m indonesian_banking_asr.synthetic.cli \
  --output-path data/synthetic/manifests/pilot.jsonl \
  --seed 123 \
  --limit 12
```

Current CLI reads:

```text
data/templates/banking_intents.yaml
```

Then writes JSONL rows with:

```text
utterance_id
template_id
intent
text
language_mix
source
generator
entities
split_group
split
```

## Verification

Run:

```bash
uv run pytest -v
```

Result:

```text
16 passed
```

CLI smoke:

```bash
uv run banking-asr-generate-text \
  --output-path "/var/folders/84/49q713852p38lz26x_hv08bm0000gn/T/opencode/banking-asr-step3.jsonl" \
  --seed 123 \
  --limit 12
```

## Current Limitations

- Only 12 templates total, one per intent.
- Gemini API not called live yet.
- No Gemini validation loop integrated with manifest generation yet.
- No audio/TTS yet.
- No template-family heldout expansion beyond deterministic split key.

## Next Step

Step 4 should add Gemini integration in safe dry-run/live modes:

- Build request wrapper around Gemini API.
- Read API key from environment, never config file.
- Generate paraphrases per canonical row.
- Validate paraphrases with existing validator.
- Write accepted/rejected paraphrase audit files.
- Keep deterministic canonical rows as fallback.

## Related

- [Step 2: Synthetic Text Generator & Validator](step-2-synthetic-text-generator-and-validator.md)
