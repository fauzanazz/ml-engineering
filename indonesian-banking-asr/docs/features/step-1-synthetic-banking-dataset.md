---
title: "Step 1: Synthetic Banking Dataset"
type: [feature-note, adr]
created: 2026-05-13
status: planned
categories: [asr, dataset, synthetic-data, banking, gemini]
related:
  - ../../README.md
author: fauzan
---

# Step 1: Synthetic Banking Dataset

Membuat dataset ASR domain perbankan Indonesia secara sintetis dengan template terstruktur, Gemini API untuk paraphrase natural, dan label entity yang deterministik.

## Spesifikasi

Tujuan minimal: dataset text-audio banking Indonesia yang cukup realistis untuk fine-tune dan evaluate Whisper-small LoRA.

Dataset harus mendukung:

- Istilah banking Indonesia: `rekening`, `cicilan`, `suku bunga`, `KPR`, `KTA`, `virtual account`, `BI-FAST`.
- Code-switching Bahasa Indonesia + English dalam satu utterance.
- Entity penting: nominal uang, nomor rekening, product name, tanggal jatuh tempo, interest rate.
- Simulasi audio call: G.711, room impulse response, office/call-center background noise.
- Entity-aware evaluation: error pada entity bernilai lebih mahal dari word biasa.

## Keputusan Dataset

### Synthetic-first, bukan cari real banking calls dulu

**Keputusan**: Step 1 memakai synthetic dataset berbasis template + Gemini API.

**Alasan**:

- Real banking support-call data punya risiko privacy, PII, dan compliance.
- Dataset ASR Indonesia publik yang ditemukan tidak domain-specific banking.
- Success metric project bergantung pada entity dan istilah banking, bukan general Indonesian WER.
- Synthetic data memberi kontrol pada distribusi entity, product, intent, dan code-switch.

### Template sebagai source of truth

**Keputusan**: entity dibuat deterministik dari template, bukan dibuat bebas oleh Gemini.

**Alasan**: LLM bisa mengganti nominal, nomor rekening, atau product name. Untuk ASR, ground truth harus pasti. Gemini hanya boleh memperkaya phrasing.

Flow:

```text
template
  -> sample entity values
  -> canonical utterance
  -> Gemini paraphrases
  -> entity preservation validator
  -> accepted dataset rows
```

### Gemini sebagai paraphraser, bukan labeler

**Keputusan**: Gemini API dipakai untuk naturalisasi bahasa, casual spoken style, dan code-switch variants.

**Alasan**: LLM bagus untuk variasi phrasing, tapi tidak boleh jadi authority untuk label penting.

Gemini output ditolak jika:

- Entity wajib hilang.
- Entity berubah.
- Nomor baru muncul.
- Nominal baru muncul.
- Meaning berubah.
- Output bukan JSON valid.
- Utterance terlalu panjang atau tidak natural.

## Referensi Riset

### FraudZen Dataset

Reference:

- [FraudZen Dataset: Realistic Ground Truth CDRs of Bypass Fraud Techniques in Mobile Networks](https://zenodo.org/records/15706356)

Pelajaran yang dipakai:

- Synthetic data berguna ketika real fraud data sulit diperoleh.
- Ground truth perlu dikontrol dari generator, bukan hasil anotasi ambigu.
- Dataset harus menyimpan metadata generation agar reproducible.
- Realism perlu dibangun lewat scenario/behavior pattern, bukan random noise saja.

Adaptasi ke project ASR:

- Template banking intent menggantikan fraud call-detail scenario.
- Entity labels dibuat langsung dari generator.
- Manifest menyimpan `template_id`, `intent`, `entities`, `generator_version`, dan augmentation metadata.

### TeleAntiFraud-28k

Reference:

- [TeleAntiFraud-28k: An Audio-Text Slow-Thinking Dataset for Telecom Fraud Detection](https://arxiv.org/html/2503.24115v2)

Pelajaran yang dipakai:

- Audio-text dataset perlu pasangan audio, transcript, dan label reasoning/task metadata.
- Telecom/call domain butuh variasi conversational style.
- Synthetic or curated audio-text data bisa dipakai untuk domain-specific downstream task.

Adaptasi ke project ASR:

- Fokus bukan fraud reasoning, tapi transcription correctness.
- Dataset tetap menyimpan rich metadata agar bisa evaluate per-intent dan per-entity.
- Call degradation dibuat eksplisit karena support-call audio berbeda dari clean read speech.

## Intent Awal

Step 1 memakai 12 intent awal:

| Intent | Contoh |
|---|---|
| `check_balance` | Saya mau cek saldo rekening saya. |
| `check_transaction_history` | Bisa bantu cek mutasi rekening bulan ini? |
| `check_installment` | Saya mau cek cicilan kartu kredit saya. |
| `credit_card_limit` | Limit kartu kredit saya tinggal berapa? |
| `loan_interest_rate` | Berapa suku bunga pinjaman saya sekarang? |
| `mortgage_interest_rate` | Interest rate KPR saya fixed atau floating? |
| `account_blocked` | Rekening saya terblokir setelah salah PIN. |
| `card_blocked` | Kartu debit saya blocked dan tidak bisa dipakai. |
| `transfer_failed` | Transfer BI-FAST saya gagal tapi saldo terpotong. |
| `virtual_account_payment` | Pembayaran virtual account saya belum masuk. |
| `change_phone_number` | Saya mau ganti nomor HP mobile banking. |
| `complaint_fee_charge` | Saya mau komplain biaya admin yang terdebit. |

## Entity Schema

Minimum entity:

| Entity | Example | Match rule |
|---|---|---|
| `ACCOUNT_NUMBER` | `1234567890` | exact digit sequence |
| `AMOUNT` | `Rp1.250.000` | normalized numeric exact match |
| `PRODUCT_NAME` | `kartu kredit` | exact or alias match |
| `BANKING_TERM` | `cicilan` | exact normalized term |
| `INTEREST_RATE` | `7,5%` | normalized percentage exact match |
| `DATE` | `15 Mei 2026` | normalized date match |
| `CARD_LAST4` | `7789` | exact 4 digits |
| `TENOR` | `24 bulan` | normalized duration match |
| `TRANSFER_METHOD` | `BI-FAST` | exact or alias match |
| `MERCHANT_NAME` | `Tokopedia` | exact normalized string |

## Template Example

Template file target:

```text
data/templates/banking_intents.yaml
```

Example:

```yaml
- template_id: check_installment_001
  intent: check_installment
  text: "Saya mau cek cicilan {product_name} saya bulan ini sebesar {amount}."
  entities:
    - type: BANKING_TERM
      slot: cicilan
      value: cicilan
    - type: PRODUCT_NAME
      slot: product_name
    - type: AMOUNT
      slot: amount
  code_switch_allowed: true
```

Filled canonical utterance:

```text
Saya mau cek cicilan kartu kredit saya bulan ini sebesar Rp1.250.000.
```

Entity labels:

```json
[
  {"type": "BANKING_TERM", "text": "cicilan", "start_char": 14, "end_char": 21},
  {"type": "PRODUCT_NAME", "text": "kartu kredit", "start_char": 22, "end_char": 34},
  {"type": "AMOUNT", "text": "Rp1.250.000", "start_char": 58, "end_char": 70}
]
```

## Gemini Prompt Contract

Gemini prompt must constrain entity preservation:

```text
Rewrite this Indonesian banking call-center utterance into 5 natural variants.

Rules:
- Keep exact entity strings unchanged.
- Do not change numbers.
- Do not add new account numbers.
- Do not add new money amounts.
- Keep same meaning.
- Include casual spoken Indonesian.
- Include 1 variant with Indonesian-English code-switching.
- Output JSON array only.

Input:
"Saya mau cek cicilan kartu kredit saya bulan ini sebesar Rp1.250.000."

Entities that must stay exact:
- cicilan
- kartu kredit
- Rp1.250.000
```

Accepted example:

```json
[
  "Saya mau tanya cicilan kartu kredit saya bulan ini yang Rp1.250.000.",
  "Bisa cek cicilan kartu kredit saya untuk bulan ini Rp1.250.000?",
  "Mau konfirmasi cicilan kartu kredit saya, nominalnya Rp1.250.000 bulan ini.",
  "Saya ingin check cicilan kartu kredit saya bulan ini, amount-nya Rp1.250.000.",
  "Tolong bantu lihat cicilan kartu kredit saya bulan ini sebesar Rp1.250.000."
]
```

## Manifest Format

Target output:

```text
data/synthetic/manifests/banking_synthetic_text_v1.jsonl
```

One row:

```json
{
  "utterance_id": "syn_id_check_installment_001_000001_p03",
  "template_id": "check_installment_001",
  "intent": "check_installment",
  "text": "Saya ingin check cicilan kartu kredit saya bulan ini, amount-nya Rp1.250.000.",
  "language_mix": "id-en",
  "source": "template_gemini",
  "generator": {
    "template_version": "v1",
    "llm": "gemini",
    "prompt_version": "v1"
  },
  "entities": [
    {"type": "BANKING_TERM", "text": "cicilan", "start_char": 18, "end_char": 25},
    {"type": "PRODUCT_NAME", "text": "kartu kredit", "start_char": 26, "end_char": 38},
    {"type": "AMOUNT", "text": "Rp1.250.000", "start_char": 72, "end_char": 84}
  ],
  "split_group": "check_installment_001"
}
```

After TTS and augmentation, add:

```json
{
  "audio_path": "data/synthetic/audio/augmented/syn_id_check_installment_001_000001_p03_aug01.wav",
  "clean_audio_path": "data/synthetic/audio/clean/syn_id_check_installment_001_000001_p03.wav",
  "duration_sec": 5.42,
  "tts": {
    "provider": "google_cloud_tts",
    "voice": "id-ID-Standard-A",
    "speaking_rate": 1.0
  },
  "augmentation": {
    "codec": "g711_ulaw",
    "rir": "small_office_01",
    "noise": "office_keyboard_02",
    "snr_db": 15
  }
}
```

## Dataset Size Target

Pilot:

```text
12 intents
20 templates / intent
5 paraphrases / template
= 1,200 text utterances
```

MVP:

```text
12 intents
300 seed utterances / intent
5 paraphrases / seed
= 18,000 text utterances
```

Audio MVP:

```text
18,000 clean TTS files
2 augmented copies / clean file
= 54,000 total audio rows including clean
```

## Split Strategy

Do not random-split final rows only.

Split by:

```text
template_id
account_number bucket
amount bucket
```

Rules:

- Held-out `template_id` family cannot appear in train.
- Held-out account numbers cannot appear in train.
- Held-out amount values cannot appear in train.
- Validation/test must be balanced by intent and entity type.

Target:

```text
train: 80%
validation: 10%
test: 10%
```

## Audio Augmentation Plan

Clean TTS is too easy. Every sample should get call-like variants.

Augmentations:

```text
resample 16 kHz mono
speed perturb 0.95-1.05
volume randomization
bandpass 300-3400 Hz
G.711 μ-law / A-law encode-decode
RIR convolution
office/call-center background noise
SNR randomization 5-25 dB
```

Noise/RIR references:

- MUSAN: `https://www.openslr.org/17/`
- Room Impulse Response and Noise Database: `https://www.openslr.org/28/`
- Simulated RIR: `https://www.openslr.org/26/`
- Aachen RIR: `https://www.openslr.org/20/`

## Evaluation Requirements

Step 1 dataset must make these metrics possible:

- Normal WER.
- Banking-term WER.
- Entity-aware WER with 3x entity penalty.
- `AMOUNT` exact-match error rate.
- `ACCOUNT_NUMBER` exact-match error rate.
- Product-name error rate.
- Code-switch WER.
- Noisy-call WER.

Entity-aware WER weighting:

```text
normal word error weight = 1
entity word error weight = 3
```

Success metric later:

```text
>15% relative WER reduction on banking terms vs baseline Whisper
<5% entity error rate on AMOUNT and ACCOUNT_NUMBER
```

## Risks

### Synthetic language too templated

Mitigation:

- Gemini paraphrases.
- Multiple prompt styles.
- Code-switch ratio control.
- Mix 10-20% general Indonesian ASR data later.
- Hold out template families.

### Gemini changes critical values

Mitigation:

- Deterministic entity source.
- Exact entity validator.
- Reject outputs with extra numbers or missing entity.
- Never use Gemini-generated labels directly.

### TTS too clean

Mitigation:

- Multiple voices.
- Speaking-rate variation.
- G.711 codec simulation.
- Bandpass filtering.
- RIR and office noise.

### Account numbers remain hard

Mitigation:

- Generate many digit styles.
- Include spoken and numeric forms.
- Evaluate account-number exact match separately.
- Avoid post-ASR correction that invents digits.

## Step 1 Exit Criteria

Step 1 complete when docs and design are clear enough to implement:

- Dataset strategy documented.
- Intent list defined.
- Entity schema defined.
- Gemini prompt contract defined.
- Manifest format defined.
- Split/leakage rules defined.
- Evaluation requirements defined.

Next step:

```text
Step 2: Implement synthetic text generator + Gemini paraphrase validator
```

## Related

- [Project README](../../README.md)
- [FraudZen Dataset](https://zenodo.org/records/15706356)
- [TeleAntiFraud-28k](https://arxiv.org/html/2503.24115v2)
