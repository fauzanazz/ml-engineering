---
title: "Step 2: Synthetic Text Generator & Validator"
type: [feature-note, implementation]
created: 2026-05-13
status: completed
categories: [asr, dataset, synthetic-data, validation, tdd]
related:
  - step-1-synthetic-banking-dataset.md
  - ../../README.md
author: fauzan
---

# Step 2: Synthetic Text Generator & Validator

Implementasi awal generator text manifest untuk synthetic Indonesian banking ASR dataset. Step ini belum memanggil Gemini API langsung; fokusnya membangun core deterministic generator, validator, dan manifest contract yang nanti dipakai oleh Gemini paraphrase layer.

## Spesifikasi

Tujuan minimal:

- Render banking template menjadi utterance canonical.
- Label entity span (`start_char`, `end_char`) secara deterministic.
- Validate Gemini/paraphrase candidates agar entity wajib tetap exact.
- Reject paraphrase dengan extra numeric token.
- Build JSONL manifest row dengan metadata generation.
- Sediakan CLI pilot untuk generate satu manifest sample.

## Keputusan Implementasi

### TDD dulu

**Keputusan**: test dibuat sebelum production code.

**Alasan**: dataset synthetic bergantung pada correctness label. Jika entity span atau validator salah, training/evaluation berikutnya akan misleading.

Tests:

```text
tests/synthetic/test_generator.py
tests/synthetic/test_validation.py
tests/synthetic/test_manifest.py
tests/synthetic/test_cli.py
```

### Generator kecil dulu

**Keputusan**: Step 2 hanya implement minimal generator, bukan full YAML loader, Gemini client, atau TTS.

**Alasan**: source-of-truth behavior harus stabil dulu: render template, label entity, validate paraphrase, write manifest.

### Entity exact preservation

**Keputusan**: paraphrase valid hanya jika semua required entity muncul exact string.

**Alasan**: nominal uang dan nomor rekening tidak boleh berubah oleh LLM.

Rejected examples:

```text
cicilan -> angsuran
kartu kredit -> kartu debit
Rp1.250.000 -> Rp2.000.000
1234567890 + extra 9876543210
```

## Files Created

```text
indonesian-banking-asr/
├── pyproject.toml
├── src/
│   └── indonesian_banking_asr/
│       ├── __init__.py
│       └── synthetic/
│           ├── __init__.py
│           ├── cli.py
│           ├── generator.py
│           ├── manifest.py
│           └── validation.py
└── tests/
    └── synthetic/
        ├── test_cli.py
        ├── test_generator.py
        ├── test_manifest.py
        └── test_validation.py
```

## Current API

### Template render

```python
from indonesian_banking_asr.synthetic.generator import EntitySpec, TemplateSpec, render_template

template = TemplateSpec(
    template_id="check_installment_001",
    intent="check_installment",
    text="Saya mau cek cicilan {product_name} sebesar {amount}.",
    entities=(
        EntitySpec(type="BANKING_TERM", slot="cicilan", value="cicilan"),
        EntitySpec(type="PRODUCT_NAME", slot="product_name"),
        EntitySpec(type="AMOUNT", slot="amount"),
    ),
)

rendered = render_template(
    template,
    values={"product_name": "kartu kredit", "amount": "Rp1.250.000"},
)
```

Output text:

```text
Saya mau cek cicilan kartu kredit sebesar Rp1.250.000.
```

Entity labels:

```json
[
  {"type": "BANKING_TERM", "text": "cicilan", "start_char": 13, "end_char": 20},
  {"type": "PRODUCT_NAME", "text": "kartu kredit", "start_char": 21, "end_char": 33},
  {"type": "AMOUNT", "text": "Rp1.250.000", "start_char": 42, "end_char": 53}
]
```

### Paraphrase validation

```python
from indonesian_banking_asr.synthetic.validation import validate_paraphrases

accepted, rejected = validate_paraphrases(
    ["Saya ingin check cicilan kartu kredit amount-nya Rp1.250.000."],
    required_entities=["cicilan", "kartu kredit", "Rp1.250.000"],
    source_text="Saya mau cek cicilan kartu kredit sebesar Rp1.250.000.",
)
```

Validation rules:

- Required entity must exist exact.
- Numeric tokens cannot appear unless present in source text.

### Manifest row

```python
from indonesian_banking_asr.synthetic.manifest import build_manifest_row

row = build_manifest_row(
    rendered,
    utterance_id="syn_id_check_installment_001_000001_p00",
    language_mix="id",
    source="template",
)
```

Manifest includes:

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
```

## CLI

Pilot command:

```bash
uv run banking-asr-generate-text --output-path data/synthetic/manifests/pilot.jsonl
```

Equivalent module command:

```bash
uv run python -m indonesian_banking_asr.synthetic.cli --output-path data/synthetic/manifests/pilot.jsonl
```

Current output: one canonical `check_installment` sample row. This is a smoke-test CLI, not full dataset generator yet.

## Verification

Run:

```bash
uv run pytest tests/synthetic -v
```

Result:

```text
7 passed
```

## Current Limitations

- No YAML template loader yet.
- No Gemini API client yet.
- No prompt builder yet.
- No split assignment yet.
- No TTS/audio generation yet.
- CLI emits one pilot row only.

## Next Step

Step 3 should add:

- Template catalog file.
- YAML or JSON template loader.
- Deterministic entity sampler.
- Multi-row manifest generation.
- Gemini prompt builder/client wrapper with dry-run mode.

## Related

- [Step 1: Synthetic Banking Dataset](step-1-synthetic-banking-dataset.md)
