# Data

Folder ini menyimpan dataset lokal untuk fraud detection pipeline.

## Dataset

Dataset utama:

```text
creditcard.csv
```

Source:

- **Name:** Credit Card Fraud Detection
- **Provider:** Machine Learning Group - ULB
- **Source:** Kaggle
- **URL:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Dataset berisi transaksi kartu kredit Eropa dari September 2013. Data sudah anonymized untuk privasi.

## Schema

Kolom:

| Column | Description |
|---|---|
| `Time` | Detik sejak transaksi pertama dalam dataset |
| `V1`–`V28` | Feature hasil PCA / anonymized |
| `Amount` | Nilai transaksi |
| `Class` | Target label: `0 = legitimate`, `1 = fraud` |

Class distribution:

```text
492 fraud / 284,807 total rows
~0.17% fraud rate
```

Dataset sangat imbalance. Karena itu project fokus ke recall, false negative, precision, F1, PR AUC, ROC AUC, threshold tuning, dan latency; bukan accuracy saja.

## Setup

1. Download dataset dari Kaggle:

   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

2. Extract file:

   ```text
   creditcard.csv
   ```

3. Simpan di folder ini:

   ```text
   fraud-detection-pipeline/data/creditcard.csv
   ```

4. Jalankan smoke test training:

   ```bash
   cd fraud-detection-pipeline
   uv run fraud-detect-train --data-path data/creditcard.csv --batch-size 10000 --model lightgbm --no-artifacts
   ```

## Git policy

Raw CSV tidak dicommit.

`.gitignore` menjaga file ini tetap lokal:

```text
data/*.csv
```

Commit yang boleh masuk dari folder ini:

```text
data/.gitkeep
data/README.md
```

## Related docs

- [Project README](../README.md)
