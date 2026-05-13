---
title: "Step 4: Gemini Paraphrase Integration & Audit Outputs"
type: [feature-note, implementation]
created: 2026-05-13
status: completed
categories: [asr, synthetic-data, gemini, audit, validation]
related:
  - step-3-yaml-catalog-entity-sampler-gemini-prompt.md
  - ../../README.md
author: fauzan
---

# Step 4: Gemini Paraphrase Integration & Audit Outputs

Step ini menghubungkan canonical synthetic manifest dengan Gemini paraphrase flow. Gemini dipakai sebagai paraphraser, lalu output divalidasi dengan entity-preservation rules sebelum diterima ke manifest.

## Spesifikasi

Tujuan minimal:

- Load Gemini config dari environment.
- Default model: `gemini-2.5-flash`.
- Jangan expose API key lewat repr/log.
- Support dry-run paraphrase mode.
- Support live Gemini paraphrase mode.
- Validate Gemini output dengan existing validator.
- Write accepted and rejected audit JSONL.

## Environment

Local secret file:

```text
.env
```

Example file committed:

```text
.env.example
```

Required:

```bash
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash
```

Rules:

- `.env` is ignored.
- `.env.example` is safe to commit.
- API key read from `os.environ`, not config file.

## Files Added

```text
src/indonesian_banking_asr/synthetic/audit.py
src/indonesian_banking_asr/synthetic/paraphrase.py
tests/synthetic/test_audit.py
tests/synthetic/test_dry_run_paraphraser.py
tests/synthetic/test_gemini_client.py
tests/synthetic/test_gemini_config.py
tests/synthetic/test_gemini_markdown.py
tests/synthetic/test_gemini_partial_fence.py
tests/synthetic/test_paraphrase.py
```

Updated:

```text
src/indonesian_banking_asr/synthetic/cli.py
src/indonesian_banking_asr/synthetic/gemini.py
```

## Gemini Client

`GeminiConfig` loads:

```python
GEMINI_API_KEY
GEMINI_MODEL
```

Default model:

```text
gemini-2.5-flash
```

`GeminiClient.generate_paraphrases(prompt)` calls:

```text
https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
```

Response parser accepts:

- raw JSON array
- fenced markdown JSON array
- partial fenced JSON array without closing fence

Reason: live Gemini sometimes returns markdown fences even when prompt says JSON only.

## Paraphrase Flow

```text
canonical rows
  -> build prompt from row text + entity strings
  -> Gemini / dry-run paraphraser
  -> parse JSON array
  -> validate exact entity preservation
  -> relabel entity spans on accepted variants
  -> write accepted JSONL
  -> write rejected audit JSONL
```

Accepted row changes:

```text
source = template_gemini
generator.llm = gemini
generator.prompt_version = v1
utterance_id p00 -> p01, p02, ...
```

Rejected audit row:

```json
{
  "source_utterance_id": "syn_id_check_installment_001_000001_p00",
  "text": "Saya mau tanya angsuran kartu kredit sebesar Rp1.250.000.",
  "reason": "missing required entity: cicilan"
}
```

## CLI

Canonical only:

```bash
uv run banking-asr-generate-text \
  --output-path data/synthetic/manifests/canonical.jsonl \
  --seed 123 \
  --limit 12
```

Dry-run paraphrase:

```bash
uv run banking-asr-generate-text \
  --output-path data/synthetic/manifests/canonical.jsonl \
  --accepted-output-path data/synthetic/manifests/accepted.jsonl \
  --rejected-output-path data/synthetic/manifests/rejected.jsonl \
  --paraphrase-mode dry-run \
  --seed 123 \
  --limit 12 \
  --variant-count 2
```

Live Gemini paraphrase:

```bash
set -a; source .env; set +a

uv run banking-asr-generate-text \
  --output-path data/synthetic/manifests/canonical.jsonl \
  --accepted-output-path data/synthetic/manifests/accepted.jsonl \
  --rejected-output-path data/synthetic/manifests/rejected.jsonl \
  --paraphrase-mode live \
  --seed 123 \
  --limit 1 \
  --variant-count 2
```

## Verification

Unit tests:

```bash
uv run pytest -v
```

Result:

```text
25 passed
```

Live smoke test ran with local `.env`:

```text
canonical: 1 rows
accepted: 2 rows
rejected: 1 rows
```

The rejected row is expected: Gemini changed/missed at least one required entity, and validator caught it.

## Current Limitations

- No retry/backoff for Gemini HTTP 429/5xx yet.
- No batching/rate-limit handling yet.
- No dotenv auto-load; shell must source `.env` manually.
- Live CLI exits on malformed/unparseable Gemini response.
- No per-row prompt/response raw audit yet.

## Next Step

Step 5 should add robustness around live generation:

- Retry with exponential backoff.
- Raw response audit for debugging.
- Continue-on-error per row.
- Configurable accepted/rejected/raw output paths.
- Larger template expansion after live behavior stabilizes.

## Related

- [Step 3: YAML Catalog, Entity Sampler & Gemini Prompt Test](step-3-yaml-catalog-entity-sampler-gemini-prompt.md)
