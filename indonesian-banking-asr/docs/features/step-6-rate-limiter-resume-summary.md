---
title: "Step 6: Rate Limiter, Resume & Batch Summary"
type: [feature-note, implementation]
created: 2026-05-13
status: completed
categories: [asr, synthetic-data, gemini, rate-limit, resume, summary]
related:
  - step-5-gemini-retry-continue-raw-audit.md
  - ../../README.md
author: fauzan
---

# Step 6: Rate Limiter, Resume & Batch Summary

Step ini menambahkan kontrol batch generation agar synthetic paraphrase generation lebih aman saat skala naik: rate limiter, resume dari output sebelumnya, dan summary report.

## Spesifikasi

Tujuan minimal:

- Rate limiter antar Gemini requests.
- Resume mode untuk skip source rows yang sudah diproses.
- Batch summary JSONL.
- CLI flags untuk rate limit, resume, dan summary output.

## Files Added

```text
src/indonesian_banking_asr/synthetic/rate_limit.py
src/indonesian_banking_asr/synthetic/resume.py
src/indonesian_banking_asr/synthetic/summary.py
tests/synthetic/test_rate_limit.py
tests/synthetic/test_resume.py
tests/synthetic/test_summary.py
```

Updated:

```text
src/indonesian_banking_asr/synthetic/cli.py
src/indonesian_banking_asr/synthetic/paraphrase.py
README.md
```

## Rate Limiter

`RateLimiter` throttles requests:

```python
RateLimiter(seconds_per_request=2.5)
```

CLI flag:

```bash
--seconds-per-request 2.5
```

Behavior:

```text
first request: no real wait
next requests: sleep seconds_per_request
```

Used via `RateLimitedParaphraser` wrapper.

## Resume Mode

Resume reads processed source IDs from prior outputs:

```text
accepted JSONL
rejected JSONL
raw JSONL
```

Supported fields:

```text
source_utterance_id
utterance_id
```

For accepted variants, source row ID is normalized:

```text
row-2_p01 -> row-2
```

CLI flag:

```bash
--resume
```

Effect:

```text
canonical rows -> remove already processed source rows -> paraphrase only pending rows
```

## Batch Summary

Summary includes:

```text
canonical_rows
pending_rows
skipped_rows
accepted_rows
rejected_rows
raw_rows
raw_status_counts
split_counts
```

CLI flag:

```bash
--summary-output-path data/synthetic/manifests/summary.jsonl
```

Example summary row:

```json
{
  "canonical_rows": 3,
  "pending_rows": 3,
  "skipped_rows": 0,
  "accepted_rows": 3,
  "rejected_rows": 1,
  "raw_rows": 3,
  "raw_status_counts": {"ok": 3},
  "split_counts": {"train": 2, "test": 1}
}
```

## CLI

Dry-run batch with summary:

```bash
uv run banking-asr-generate-text \
  --output-path data/synthetic/manifests/canonical.jsonl \
  --accepted-output-path data/synthetic/manifests/accepted.jsonl \
  --rejected-output-path data/synthetic/manifests/rejected.jsonl \
  --raw-output-path data/synthetic/manifests/raw.jsonl \
  --summary-output-path data/synthetic/manifests/summary.jsonl \
  --paraphrase-mode dry-run \
  --continue-on-error \
  --seed 123 \
  --limit 3 \
  --variant-count 1
```

Live Gemini batch with resume + rate limit:

```bash
set -a; source .env; set +a

uv run banking-asr-generate-text \
  --output-path data/synthetic/manifests/canonical.jsonl \
  --accepted-output-path data/synthetic/manifests/accepted.jsonl \
  --rejected-output-path data/synthetic/manifests/rejected.jsonl \
  --raw-output-path data/synthetic/manifests/raw.jsonl \
  --summary-output-path data/synthetic/manifests/summary.jsonl \
  --paraphrase-mode live \
  --continue-on-error \
  --resume \
  --seconds-per-request 2.5 \
  --seed 123 \
  --limit 100 \
  --variant-count 3
```

## Verification

Unit tests:

```bash
uv run pytest -v
```

Result:

```text
31 passed
```

Dry-run smoke:

```text
step6-canonical.jsonl: 3 rows
step6-accepted.jsonl: 3 rows
step6-rejected.jsonl: 1 rows
step6-raw.jsonl: 3 rows
step6-summary.jsonl: 1 rows
```

## Current Limitations

- Resume rewrites output files instead of append/merge.
- No persistent job manifest/checkpoint file yet.
- Rate limiter is fixed delay, not adaptive from Gemini quota headers.
- Summary is one JSONL row, not a human-readable report table.

## Next Step

Step 7 should add dataset expansion controls:

- More templates per intent.
- Config-driven generation counts.
- Append-safe output writer.
- Human review sample export.

## Related

- [Step 5: Gemini Retry, Continue-on-Error & Raw Audit](step-5-gemini-retry-continue-raw-audit.md)
