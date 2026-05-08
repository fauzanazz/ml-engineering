---
title: "Step 1: Project Scaffold & One-Batch Baseline"
type: [feature-note, adr]
created: 2026-05-08
status: completed
categories: [ml-pipeline, scaffold, baseline, cli]
related:
  - ../plans/2026-05-08-minimal-fraud-detection-pipeline.md
  - ../plans/2026-05-08-minimal-fraud-detection-design.md
  - step-2-training-evaluation.md
author: fauzan
---

# Step 1: Project Scaffold & One-Batch Baseline

Setup proyek ML Engineering yang benar dari nol — struktur package Python, tooling, dan satu run baseline untuk buktikan pipeline hidup.

## Spesifikasi

Tujuan minimal: fraud detection pipeline sebagai kendaraan belajar MLE "cara yang benar". Bukan riset akurasi, tapi praktik struktur kode, tooling, dan testability yang proper.

## Dataset

[Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (`mlg-ulb/creditcardfraud`).

- Simpan lokal: `data/creditcard.csv`
- 284.807 transaksi, fitur PCA `V1`–`V28` + `Time` + `Amount`, label `Class` (0=legit, 1=fraud)
- ~0.17% fraud — sangat imbalanced

Download manual dari Kaggle lalu taruh di `data/`. File ini di-gitignore.

## Keputusan Scaffold

### uv sebagai package manager

**Keputusan**: `uv` — bukan pip, poetry, atau conda.

**Alasan**: uv lebih cepat, lockfile deterministik, `uv run` tanpa aktivasi venv manual. Standar modern Python tooling.

### Python package proper, bukan notebook

**Keputusan**: `src/fraud_detection/` sebagai installable package, bukan Jupyter notebook atau script satu file.

**Alasan**: belajar MLE yang benar. Notebook tidak testable, tidak maintainable di produksi. Package bisa ditest per-modul, di-import, dan di-extend tanpa spaghetti.

### Nama package & CLI

- Package: `fraud_detection`
- CLI entry point: `fraud-detect-train` (didefinisikan di `pyproject.toml` `[project.scripts]`)

### LightGBM factory adapter

**Keputusan**: factory function `build_lgbm_classifier()` dari awal, bukan inisiasi model langsung di training loop.

**Alasan**: isolasi konfigurasi model dari training logic. Mudah swap model lain tanpa ubah pipeline. YAGNI — tapi factory lebih murah dari refactor nanti.

## File yang Dibuat di Scaffold

```
fraud-detection-pipeline/
├── pyproject.toml              # package config, dependencies, scripts
├── uv.lock                     # lockfile deterministik
├── src/
│   └── fraud_detection/
│       ├── __init__.py
│       ├── data.py             # load_data(path, batch_size)
│       ├── models.py           # build_lgbm_classifier()
│       ├── training.py         # train_model(X, y)
│       ├── metrics.py          # compute_metrics(y_true, y_pred, y_proba)
│       └── cli.py              # entry point fraud-detect-train
├── tests/
│   ├── test_data.py
│   ├── test_models.py
│   ├── test_training.py
│   ├── test_metrics.py
│   └── test_cli.py
└── data/                       # gitignored, taruh creditcard.csv di sini
```

## Baseline: Train & Evaluate Satu Batch

Step 1 sengaja sederhana: load satu batch, train, evaluate — semua pada **batch yang sama** (bukan split). Tujuannya hanya buktikan pipeline tidak error dari ujung ke ujung.

### Perintah

```bash
# Install dependencies
uv sync

# Jalankan test
uv run pytest -v

# Run baseline (256 baris pertama)
uv run fraud-detect-train --data-path data/creditcard.csv --batch-size 256
```

### Masalah yang Ditemukan di Run Pertama

Run pertama dengan `--batch-size 256` menghasilkan akurasi 100% — tampak sempurna, tapi **menyesatkan**.

**Penyebab**: 256 baris pertama dataset semua label `Class=0` (transaksi normal). Model trivially predict semua normal → akurasi 100%, tapi tidak pernah prediksi fraud satu pun.

**Pelajaran**:
1. Accuracy bukan metrik yang tepat untuk dataset imbalanced.
2. Evaluate pada data yang sama dengan train tidak bermakna.
3. Perlu split berbasis waktu agar evaluasi jujur.

Ini bukan bug — ini signal bahwa step 1 perlu dilanjutkan ke step 2.

## Transisi ke Step 2

Step 1 selesai ketika:
- `uv run pytest -v` hijau semua
- CLI bisa dijalankan tanpa error

Tapi hasilnya belum bermakna. Step 2 menambahkan:
- **Time-based split**: 80% train / 20% test, sort by `Time`
- **Metrik proper**: precision, recall, F1, PR-AUC — bukan accuracy
- **Batch default lebih besar**: 10.000 baris agar ada fraud di sample

→ Lihat [Step 2: Training & Evaluation Pipeline](step-2-training-evaluation.md)

## Related

- [Step 2: Training & Evaluation Pipeline](step-2-training-evaluation.md)
- [Design Doc](../plans/2026-05-08-minimal-fraud-detection-design.md)
- [Pipeline Plan](../plans/2026-05-08-minimal-fraud-detection-pipeline.md)
