---
title: "Step 2: Training & Evaluation Pipeline"
type: [feature-note, adr]
created: 2026-05-09
status: completed
categories: [ml-pipeline, training, evaluation, cli]
related:
  - step-1-project-scaffold-and-one-batch-baseline.md
  - ../plans/2026-05-08-minimal-fraud-detection-pipeline.md
  - ../plans/2026-05-08-minimal-fraud-detection-design.md
author: fauzan
---

# Step 2: Training & Evaluation Pipeline

Membangun pipeline training dan evaluasi fraud detection end-to-end — dari load CSV sampai cetak metrik di terminal — menggunakan LightGBM, time-based split, dan CLI satu perintah.

## Konteks

Lanjutan dari [Step 1: Project Scaffold & One-Batch Baseline](step-1-project-scaffold-and-one-batch-baseline.md).

Step 1 membuktikan pipeline hidup tapi evaluasi tidak bermakna: 256 baris pertama semua normal, accuracy 100% menyesatkan. Step 2 memperbaiki dua hal: split berbasis waktu dan metrik yang jujur untuk dataset imbalanced.

Proyek ini adalah studi ML Engineering "cara yang benar": package Python proper, tidak ada notebook, tidak ada script satu file. `uv`, `pyproject.toml`, struktur direktori, dan pytest sudah berjalan sejak step 1.

Dataset: [Kaggle Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (`data/creditcard.csv`), 284.807 transaksi, fitur PCA `V1`–`V28` + `Time` + `Amount`, label `Class` (0=legit, 1=fraud).

Tujuan step 2: buktikan seluruh pipeline bisa dijalankan dengan satu perintah, semua tertest, tidak ada magic number tersebar.

## Cara Kerja

Pipeline mengalir linear: load → split → train → predict → metrics → print.

**Load** (`data.py`): baca CSV dengan `pandas`, ambil `batch_size` baris pertama (default 10.000 untuk iterasi cepat). Feature engineering nol — data sudah PCA.

**Split** (`training.py`): sort berdasarkan kolom `Time` lalu potong. 80% pertama jadi train, 20% terakhir jadi test. Ini _time-based split_ — meniru kondisi produksi di mana model dilatih pada transaksi lama dan diuji pada transaksi baru. Split rasio divalidasi: `0 < test_size < 1`, lempar `ValueError` jika tidak valid.

**Model** (`models.py`): factory function `build_lgbm_classifier()` mengembalikan `LGBMClassifier` dengan hyperparameter tetap (scale_pos_weight untuk imbalance, verbosity=-1). Factory pattern dipilih agar mudah swap model lain nanti tanpa ubah training logic.

**Metrics** (`metrics.py`): adapter di atas sklearn — `compute_metrics(y_true, y_pred, y_proba)` mengembalikan dict `{precision, recall, f1, pr_auc}`. PR-AUC dihitung dari `y_proba` (probabilitas kelas 1), bukan prediksi biner. `zero_division=0` mencegah warning saat precision/recall undefined.

**CLI** (`cli.py`): entry point `fraud-detect-train` (didefinisikan di `pyproject.toml`). Menerima `--data-path`, `--batch-size`, `--test-size`. Cetak semua metrik ke stdout, tidak dump raw predictions.

### Peta File

| File | Peran |
|------|-------|
| `src/fraud_detection/data.py` | Load CSV, `load_data(path, batch_size)` |
| `src/fraud_detection/models.py` | Factory `build_lgbm_classifier()` |
| `src/fraud_detection/training.py` | `time_based_split(df, test_size)`, `train_model(X, y)` |
| `src/fraud_detection/metrics.py` | `compute_metrics(y_true, y_pred, y_proba)` |
| `src/fraud_detection/cli.py` | CLI entry point, orkestrasi pipeline |
| `tests/test_data.py` | Unit test loader |
| `tests/test_models.py` | Unit test factory |
| `tests/test_training.py` | Unit test split |
| `tests/test_metrics.py` | Unit test metrics adapter |
| `tests/test_training_metrics.py` | Integration: split + train + metrics |
| `tests/test_cli.py` | CLI smoke test |

### Cara Jalankan

```bash
# Install dependencies
uv sync

# Jalankan semua test
uv run pytest -v

# Training dengan batch 10.000 (default)
uv run fraud-detect-train --data-path data/creditcard.csv

# Custom batch dan split ratio
uv run fraud-detect-train --data-path data/creditcard.csv --batch-size 50000 --test-size 0.2
```

## Keputusan & Trade-off

### Time-based split, bukan random split

**Keputusan**: sort by `Time`, ambil 80% awal untuk train, 20% akhir untuk test.

**Alasan**: fraud detection di produksi selalu prediksi masa depan dari model yang dilatih di masa lalu. Random split bocorkan informasi temporal — model bisa "lihat ke depan". Time-based split simulasikan kondisi nyata.

**Trade-off**: dataset lebih kecil jadi train dan test tidak IID, tapi ini _benar_ bukan bug.

### Factory function, bukan class

**Keputusan**: `build_lgbm_classifier()` return instance, bukan class `LGBMModelFactory`.

**Alasan**: tidak ada state yang perlu disimpan, tidak ada polymorphism runtime. Function lebih sederhana dari class. YAGNI — kalau nanti perlu multi-model, baru tambah abstraksi.

### PR-AUC sebagai metrik utama

**Keputusan**: prioritaskan PR-AUC di atas ROC-AUC.

**Alasan**: dataset sangat imbalanced (~0.17% fraud). ROC-AUC bisa tinggi palsu karena True Negative banyak. PR-AUC fokus pada kinerja di kelas minoritas (fraud), lebih jujur untuk use case ini.

### batch_size default 10.000

**Keputusan**: load hanya 10.000 baris dari 284.807 untuk iterasi cepat.

**Alasan**: step pembelajaran. Full dataset bisa dipakai dengan `--batch-size 284807`. Default kecil agar `uv run fraud-detect-train` selesai dalam detik, bukan menit.

## Bukti Verifikasi

```
$ uv run pytest -v
...
21 passed in X.XXs
```

```
$ uv run fraud-detect-train --data-path data/creditcard.csv --batch-size 10000
training_accuracy: 1.0000
test_accuracy:     0.9985
precision:         0.8125
recall:            1.0000
f1:                0.8966
pr_auc:            0.9682
```

Model recall 1.0000 artinya tidak ada fraud yang lolos (zero false negative di batch ini). Precision 0.8125 artinya ~19% false positive — masih bisa ditoleransi di step pembelajaran. PR-AUC 0.9682 menunjukkan diskriminasi yang kuat di seluruh threshold.

## Keterbatasan & Langkah Selanjutnya

- **Hyperparameter tetap** — tidak ada tuning. LightGBM default + `scale_pos_weight` saja.
- **Tidak ada feature engineering** — pakai fitur PCA mentah dari Kaggle.
- **Tidak ada model serialization** — model tidak disimpan ke disk, tidak ada serving.
- **Batch kecil default** — hasil di atas dari 10.000 baris, bukan full dataset. Overfitting bisa terjadi di training set kecil (training_accuracy=1.0).
- **Tidak ada experiment tracking** — tidak ada MLflow/W&B logging.

**Step 3 kandidat**: model serialization (simpan/load model), atau feature engineering (rolling stats, amount binning), atau hyperparameter tuning dengan optuna.

## Related

- [Step 1: Project Scaffold & One-Batch Baseline](step-1-project-scaffold-and-one-batch-baseline.md) — awal perjalanan, scaffold dan baseline menyesatkan
- [Design Doc](../plans/2026-05-08-minimal-fraud-detection-design.md) — arsitektur awal yang jadi acuan
- [Pipeline Plan](../plans/2026-05-08-minimal-fraud-detection-pipeline.md) — rencana implementasi step-by-step
- [LightGBM docs](https://lightgbm.readthedocs.io/)
- [sklearn metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
