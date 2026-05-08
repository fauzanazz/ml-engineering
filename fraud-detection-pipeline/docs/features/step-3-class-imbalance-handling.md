---
title: "Step 3 — Penanganan Ketidakseimbangan Kelas"
step: 3
status: selesai
tanggal: 2026-05-09
tags: [imbalance, scale_pos_weight, lightgbm, recall]
previous: step-2-training-evaluation.md
next: step-4-run-logging-and-artifacts.md
---

# Step 3 — Penanganan Ketidakseimbangan Kelas

## Latar Belakang

Data transaksi keuangan secara inheren sangat tidak seimbang. Pada batch latih yang digunakan, hanya **25 fraud dari 8.000 transaksi** (~0,312%, rasio ~319:1). Kondisi ini membuat model default cenderung mengabaikan kelas minoritas (fraud).

**Prioritas bisnis**: *false negative* (fraud tidak terdeteksi) jauh lebih mahal daripada *false positive* (transaksi sah yang ditandai). Oleh karena itu, model harus dioptimalkan untuk **recall tinggi** pada kelas fraud.

## Keputusan Desain

| Opsi | Dipilih? | Alasan |
|------|----------|--------|
| SMOTE / oversampling data | ❌ | Menambah kompleksitas pipeline, rentan data leakage lintas fold |
| Undersampling kelas mayoritas | ❌ | Membuang data valid yang berpotensi informatif |
| `scale_pos_weight` (algorithm-level) | ✅ | Sederhana, tidak mengubah data, dihitung langsung dari distribusi target latih |

`scale_pos_weight` dihitung sebagai:

```
scale_pos_weight = jumlah_negatif / jumlah_positif
```

Nilai ini memberi bobot lebih pada tiap sampel fraud saat optimasi loss LightGBM.

## Implementasi

### Peta File

| File | Perubahan |
|------|-----------|
| `src/fraud_detection/models.py` | `LightGbmFactory` menerima `scale_pos_weight`; factory protocol mendukung parameter override |
| `src/fraud_detection/training.py` | Fungsi `compute_scale_pos_weight`, guard jika tidak ada positif/negatif, parameter `imbalance_strategy`, custom factory tidak diabaikan |
| `src/fraud_detection/cli.py` | Flag `--imbalance-strategy` dengan pilihan `none` \| `scale-pos-weight` |
| `tests/` | Test suite diperluas; total **35 test** lulus sebelum Step 5 |

### Cara Pakai

```bash
# Tanpa penanganan imbalance (default lama)
python -m fraud_detection.cli train --imbalance-strategy none

# Dengan scale_pos_weight
python -m fraud_detection.cli train --imbalance-strategy scale-pos-weight
```

### Logika `compute_scale_pos_weight`

```python
# src/fraud_detection/training.py
def compute_scale_pos_weight(y: pd.Series) -> float:
    positive_count = y.sum()
    negative_count = len(y) - positive_count
    if positive_count == 0 or negative_count == 0:
        return 1.0
    return negative_count / positive_count
```

Guard mencegah division-by-zero dan mengembalikan `1.0` (netral) bila kelas hanya satu.

## Hasil Evaluasi

Batch 10.000 transaksi, dibandingkan dengan baseline Step 2.

| Metrik | Baseline (no weighting) | scale-pos-weight |
|--------|------------------------|-----------------|
| Training accuracy | 1.0000 | 0.9989 |
| Test accuracy | 0.9985 | 0.9970 |
| Precision | 0.8125 | 0.7059 |
| Recall | 1.0000 | 0.9231 |
| F1 | 0.8966 | 0.8000 |
| PR-AUC | 0.9682 | 0.6928 |

### Interpretasi

- `scale_pos_weight` **tidak mengalahkan baseline** pada batch sempit ini. Baseline kebetulan mencapai recall sempurna karena distribusi test set yang spesifik.
- Namun, weighting **mengkodekan preferensi bisnis** secara eksplisit: model lebih berani mengorbankan presisi demi recall tinggi.
- Penurunan PR-AUC mengindikasikan trade-off nyata yang justru membuka jalan ke **Step 5: threshold tuning** — menyesuaikan ambang keputusan secara sadar alih-alih mengandalkan default 0.5.

## Keterbatasan

- Nilai `scale_pos_weight` dihitung dari batch latih saja; tidak mencerminkan distribusi populasi penuh.
- Pada batch sangat kecil (< 50 fraud), pengaruh weighting tidak stabil.
- Tidak ada validasi silang; evaluasi dilakukan pada satu split.

## Langkah Berikutnya

→ [Step 4 — Run Logging dan Artefak](step-4-run-logging-and-artifacts.md): mencatat metrik dan model ke MLflow.  
→ [Step 5 — Threshold Tuning untuk Recall](step-5-threshold-tuning-for-recall.md): mengoptimalkan ambang klasifikasi berdasarkan kurva precision-recall.
