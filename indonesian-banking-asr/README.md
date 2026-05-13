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

Detail tiap step:

- [Step 1: Synthetic Banking Dataset](docs/features/step-1-synthetic-banking-dataset.md)

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

## References

- [FraudZen Dataset: Realistic Ground Truth CDRs of Bypass Fraud Techniques in Mobile Networks](https://zenodo.org/records/15706356)
- [TeleAntiFraud-28k: An Audio-Text Slow-Thinking Dataset for Telecom Fraud Detection](https://arxiv.org/html/2503.24115v2)
