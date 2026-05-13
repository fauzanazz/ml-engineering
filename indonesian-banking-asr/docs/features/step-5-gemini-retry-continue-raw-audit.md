---
title: "Step 5: Gemini Retry, Continue-on-Error & Raw Audit"
type: [feature-note, implementation]
created: 2026-05-13
status: completed
categories: [asr, synthetic-data, gemini, retry, audit, resilience]
related:
  - step-4-gemini-paraphrase-integration-audit.md
  - ../../README.md
author: fauzan
---

# Step 5: Gemini Retry, Continue-on-Error & Raw Audit

Step ini menutup risiko operasional dari Step 4: live Gemini batch generation tidak boleh gagal total hanya karena satu request error, response perlu diaudit, dan 429/5xx perlu retry dengan backoff.

## Spesifikasi

Tujuan minimal:

- Retry Gemini request untuk error sementara.
- Exponential backoff.
- Continue-on-error per row.
- Raw paraphrase audit JSONL.
- CLI flags untuk raw audit dan continue-on-error.

## Files Added

```text
tests/synthetic/test_gemini_retry.py
tests/synthetic/test_paraphrase_audit.py
```

Updated:

```text
src/indonesian_banking_asr/synthetic/gemini.py
src/indonesian_banking_asr/synthetic/paraphrase.py
src/indonesian_banking_asr/synthetic/cli.py
README.md
```

## Gemini Retry

`GeminiClient` now supports:

```python
GeminiClient(
    config=config,
    max_retries=3,
    sleep=time.sleep,
)
```

Retryable errors:

```text
HTTP 429
HTTP 500
HTTP 502
HTTP 503
HTTP 504
```

Backoff schedule with default `max_retries=3`:

```text
attempt 1 fail -> sleep 1s
attempt 2 fail -> sleep 2s
attempt 3 fail -> raise
```

Tests inject fake sleep to avoid slow test runtime.

## Continue-on-Error

New function:

```python
paraphrase_rows_with_audit(
    rows,
    paraphraser=paraphraser,
    variant_count=variant_count,
    continue_on_error=True,
)
```

If a row fails and `continue_on_error=True`, batch continues.

Rejected audit row for generation error:

```json
{
  "source_utterance_id": "syn_id_check_balance_001_000001_p00",
  "text": "",
  "reason": "paraphrase generation failed: Gemini temporary failure"
}
```

If `continue_on_error=False`, error is raised immediately.

## Raw Audit

Raw audit captures prompt and raw variants or error.

Success row:

```json
{
  "source_utterance_id": "syn_id_check_balance_001_000002_p00",
  "status": "ok",
  "prompt": "...",
  "raw_variants": ["..."]
}
```

Error row:

```json
{
  "source_utterance_id": "syn_id_check_balance_001_000001_p00",
  "status": "error",
  "prompt": "...",
  "error": "Gemini temporary failure"
}
```

## CLI

Live generation with resilience:

```bash
set -a; source .env; set +a

uv run banking-asr-generate-text \
  --output-path data/synthetic/manifests/canonical.jsonl \
  --accepted-output-path data/synthetic/manifests/accepted.jsonl \
  --rejected-output-path data/synthetic/manifests/rejected.jsonl \
  --raw-output-path data/synthetic/manifests/raw.jsonl \
  --paraphrase-mode live \
  --continue-on-error \
  --seed 123 \
  --limit 1 \
  --variant-count 1
```

Dry-run works with same audit flags:

```bash
uv run banking-asr-generate-text \
  --output-path data/synthetic/manifests/canonical.jsonl \
  --accepted-output-path data/synthetic/manifests/accepted.jsonl \
  --rejected-output-path data/synthetic/manifests/rejected.jsonl \
  --raw-output-path data/synthetic/manifests/raw.jsonl \
  --paraphrase-mode dry-run \
  --continue-on-error \
  --seed 123 \
  --limit 12 \
  --variant-count 1
```

## Verification

Unit tests:

```bash
uv run pytest -v
```

Result:

```text
27 passed
```

Live smoke test:

```text
step5-canonical-one.jsonl: 1 rows
step5-accepted-one.jsonl: 1 rows
step5-rejected-one.jsonl: 1 rows
step5-raw-one.jsonl: 1 rows
```

Note: one rejected row can happen even when generation succeeds, because validator rejects variants where Gemini changes or omits required entity.

## Updated Risk Status

Resolved from Step 4:

- Retry/backoff exists.
- Continue-on-error exists.
- Raw Gemini response audit exists.

Remaining risks:

- No explicit rate limiter yet.
- No resumable job state yet.
- No batch-level summary report yet.
- Raw audit stores prompts; prompts can contain synthetic account numbers, so keep raw audit under ignored `data/synthetic/` unless reviewed.

## Next Step

Step 6 should add larger dataset generation controls:

- Rate limiter.
- Batch summary metrics.
- Resume from existing output files.
- Config-driven generation counts.
- Template expansion beyond 12 templates.

## Related

- [Step 4: Gemini Paraphrase Integration & Audit Outputs](step-4-gemini-paraphrase-integration-audit.md)
