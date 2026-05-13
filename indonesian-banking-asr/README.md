# Indonesian Banking ASR

Indonesian Banking ASR adalah project eksperimen ML Engineering untuk membangun speech-to-text domain perbankan Indonesia. Fokus utama: transkripsi support call yang tahan terhadap istilah banking, code-switching Bahasa Indonesia + English, dan audio call yang noisy/degraded.

## Tujuan Project

Project ini menjawab pertanyaan utama:

- Bagaimana membuat dataset ASR banking Indonesia ketika real call-center data sulit dipakai karena privacy?
- Bagaimana memastikan istilah penting seperti `rekening`, `cicilan`, `suku bunga`, nominal uang, dan nomor rekening tidak salah transkripsi?
- Bagaimana mensimulasikan audio call center dari synthetic clean speech?
- Bagaimana mengukur WER dengan penalti lebih besar pada entity banking?

Metric utama:

- **Banking-term WER reduction** — target >15% relative reduction vs baseline Whisper.
- **Entity error rate** — target <5% untuk `AMOUNT` dan `ACCOUNT_NUMBER`.
- **Entity-aware WER** — error pada nominal, nomor rekening, dan product name diberi penalti lebih besar.

## Dataset Strategy

Keputusan Step 1: dataset dibuat secara sintetis dengan **template + Gemini API**.

Alasan:

- Dataset ASR Indonesia publik tidak spesifik domain banking.
- Real support-call banking mengandung data sensitif dan sulit dipakai sejak awal.
- Synthetic generation memberi kontrol penuh atas entity penting: nominal, nomor rekening, product name, interest rate, due date.
- Evaluation bisa dibuat entity-balanced sejak awal.

High-level flow:

```text
banking templates
  -> deterministic entity sampling
  -> Gemini paraphrase with entity preservation
  -> validation / rejection
  -> TTS generation
  -> call audio augmentation
  -> manifest with entity labels
```

## Project Steps

| Step | Tanggal | Judul | Ringkasan |
|---:|---|---|---|
| 1 | 2026-05-13 | Synthetic Banking Dataset Design | Menetapkan strategi dataset sintetis berbasis template + Gemini API, entity schema, validation rules, dan referensi riset. |
| 2 | 2026-05-13 | Synthetic Text Generator & Validator | Implementasi awal template renderer, entity span labeler, paraphrase validator, manifest builder, dan pilot CLI. |
| 3 | 2026-05-13 | YAML Catalog, Entity Sampler & Gemini Prompt Test | Tambah YAML catalog 12 intents, deterministic entity sampler, deterministic split, multi-row manifest pipeline, dan Gemini prompt/parser tests. |
| 4 | 2026-05-13 | Gemini Paraphrase Integration & Audit Outputs | Tambah Gemini env config, dry-run/live paraphrase modes, entity validation, accepted/rejected audit JSONL, dan live smoke test. |
| 5 | 2026-05-13 | Gemini Retry, Continue-on-Error & Raw Audit | Tambah retry/backoff, continue-on-error per row, raw audit JSONL, dan CLI flags untuk live batch resilience. |
| 6 | 2026-05-13 | Rate Limiter, Resume & Batch Summary | Tambah fixed-delay rate limiter, resume mode, batch summary JSONL, dan CLI flags untuk batch generation lebih aman. |

Detail tiap step:

- [Step 1: Synthetic Banking Dataset](docs/features/step-1-synthetic-banking-dataset.md)
- [Step 2: Synthetic Text Generator & Validator](docs/features/step-2-synthetic-text-generator-and-validator.md)
- [Step 3: YAML Catalog, Entity Sampler & Gemini Prompt Test](docs/features/step-3-yaml-catalog-entity-sampler-gemini-prompt.md)
- [Step 4: Gemini Paraphrase Integration & Audit Outputs](docs/features/step-4-gemini-paraphrase-integration-audit.md)
- [Step 5: Gemini Retry, Continue-on-Error & Raw Audit](docs/features/step-5-gemini-retry-continue-raw-audit.md)
- [Step 6: Rate Limiter, Resume & Batch Summary](docs/features/step-6-rate-limiter-resume-summary.md)

## Current Plan

Current planned flow:

```text
template bank utterance
  -> sample entities deterministically
  -> generate canonical text
  -> ask Gemini for paraphrases/code-switch variants
  -> validate exact entity preservation
  -> write JSONL manifest
  -> generate TTS audio
  -> apply G.711 + RIR + background noise
  -> train/evaluate Whisper-small LoRA
```

Important leakage rules:

- Test template families held out from train.
- Account numbers and amount values held out from train.
- Gemini cannot create ground-truth entities freely.
- Entity labels come from deterministic generator, not LLM output.
- Test set must include manually reviewed banking utterances.

## Known Risks

Gemini live batch generation has improved resilience after Step 5:

- Retry/backoff exists for retryable Gemini failures.
- Continue-on-error exists for live batch generation.
- Raw Gemini response audit exists.

Remaining operational risks:

- Rate limiter is fixed delay, not adaptive from Gemini quota headers.
- Resume mode skips processed rows but rewrites output files; no append-safe merge yet.
- Summary is JSONL only, not a human-readable report table yet.
- Raw audit stores prompts; keep generated audit files under ignored `data/synthetic/` unless reviewed.

## References

- [FraudZen Dataset: Realistic Ground Truth CDRs of Bypass Fraud Techniques in Mobile Networks](https://zenodo.org/records/15706356)
- [TeleAntiFraud-28k: An Audio-Text Slow-Thinking Dataset for Telecom Fraud Detection](https://arxiv.org/html/2503.24115v2)
