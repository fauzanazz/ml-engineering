# Fraud Detection Pipeline

Fraud Detection Pipeline adalah project eksperimen machine learning untuk mendeteksi transaksi fraud dari dataset kartu kredit anonymized (`data/creditcard.csv`). Pipeline ini dibuat bertahap dari baseline sederhana sampai evaluasi multi-model dengan split train/validation/test, feature engineering, threshold tuning, artifact logging, dan latency metric.

## Tujuan Project

Project ini menjawab pertanyaan utama:

- Bagaimana membangun pipeline fraud detection yang tidak leakage?
- Model mana yang paling cocok untuk fraud detection dengan data imbalance?
- Bagaimana trade-off precision, recall, F1, PR AUC, dan latency?
- Bagaimana memilih threshold tanpa nyontek test set?

Karena fraud sangat imbalance, metric utama bukan accuracy. Fokus utama:

- **Recall** — fraud yang kelewat harus minimum.
- **False Negative (FN)** — missed fraud paling mahal.
- **PR AUC** — lebih cocok dari ROC AUC untuk minority class.
- **Latency** — penting untuk online / streaming fraud prediction.

## Dataset

Dataset lokal:

```text
data/creditcard.csv
```

Kolom:

- `Time`
- `V1`–`V28` — anonymized/PCA features
- `Amount`
- `Class` — label target, `1 = fraud`, `0 = legit`

Fraud sangat jarang:

```text
492 fraud / 284,807 total rows
~0.17% fraud rate
```

## Project Steps

| Step | Tanggal | Judul | Ringkasan |
|---:|---|---|---|
| 1 | 2026-05-09 | Project scaffold + one-batch baseline | Setup pipeline awal, load batch kecil, baseline menunjukkan accuracy menipu karena all-negative prediction. |
| 2 | 2026-05-09 | Time-split training/evaluation | Split berdasarkan `Time`, evaluasi precision/recall/F1/PR AUC, mulai validasi model lebih realistis. |
| 3 | 2026-05-09 | Class imbalance handling | Tambah `scale-pos-weight` untuk menangani imbalance; recall tetap tinggi dengan trade-off precision. |
| 4 | 2026-05-09 | Run logging + artifacts | Tambah artifact writer untuk `config.json`, `metrics.json`, model artifact, dan historical run log. |
| 5 | 2026-05-08 | Threshold tuning for recall | Tambah `--decision-threshold` dan threshold sweep. Belakangan dicatat sebagai exploratory karena threshold dipilih dari test set. |
| 6 | 2026-05-09 | Feature engineering + validation split | Tambah train/validation/test split, train-only feature engineering, validation-only threshold tuning, duplicate/leakage handling. |
| 7 | 2026-05-09 | Hyperparameter tuning | Tambah randomized hyperparameter search untuk LightGBM, Random Forest, dan XGBoost; tambah ROC AUC dan single-row latency untuk evaluasi online scoring. |

Detail tiap step:

- [Step 1: Project Scaffold and One-Batch Baseline](docs/features/step-1-project-scaffold-and-one-batch-baseline.md)
- [Step 2: Training Evaluation](docs/features/step-2-training-evaluation.md)
- [Step 3: Class Imbalance Handling](docs/features/step-3-class-imbalance-handling.md)
- [Step 4: Run Logging and Artifacts](docs/features/step-4-run-logging-and-artifacts.md)
- [Step 5: Threshold Tuning for Recall](docs/features/step-5-threshold-tuning-for-recall.md)
- [Step 6: Feature Engineering & Validation Split](docs/features/step-6-feature-engineering-validation-split.md)
- [Step 7: Hyperparameter Tuning](docs/features/step-7-hyperparameter-tuning.md)
- [Historical Run Log](docs/run-log.md)
- [Artifact Security](docs/artifact-security.md)

## Current Pipeline

Current flow:

```text
raw CSV
 -> sort by Time
 -> chronological train / validation / test split
 -> remove duplicate rows per split
 -> remove exact feature overlap across splits
 -> fit FeaturePipeline on train only
 -> transform train / validation / test
 -> optionally tune hyperparameters on train-only CV
 -> train model on train
 -> tune threshold on validation
 -> evaluate once on test
 -> write artifacts
```

Important leakage rules:

- Scaler / feature pipeline fit only on train.
- Threshold selected only on validation.
- Test used only for final evaluation.
- Duplicate cleanup happens after split, with cross-split exact feature overlap removal.

## Feature Engineering

`FeaturePipeline` adds:

| Feature | Description |
|---|---|
| `log_amount_raw` | `log1p(Amount)` |
| `amount_is_zero` | binary flag for zero amount |
| `hour_of_day` | derived from `Time` |
| `day` | derived from `Time` |
| `is_night` | `hour_of_day < 6` |
| `log_amount_scaled` | train-fitted robust scaling |
| `amount_scaled` | train-fitted robust scaling |

`V1`–`V28` are kept as anonymized PCA features.

## Supported Models

CLI supports:

```text
lightgbm
logistic-regression
decision-tree
random-forest
xgboost
```

Example:

```bash
uv run fraud-detect-train \
  --data-path data/creditcard.csv \
  --batch-size 50000 \
  --imbalance-strategy scale-pos-weight \
  --val-size 0.1 \
  --threshold-objective target-recall \
  --target-recall 0.95 \
  --model random-forest
```

## Latest Model Comparison

Config:

```text
batch_size=50000
imbalance_strategy=scale-pos-weight
val_size=0.1
threshold_objective=target-recall
target_recall=0.95
```

| Model | Precision | Recall | F1 | PR AUC | Batch latency | Per-row latency |
|---|---:|---:|---:|---:|---:|---:|
| LightGBM | 0.2429 | 0.9773 | 0.3891 | 0.3390 | 0.002021s | 0.000000203s |
| Logistic Regression | 0.6613 | 0.9318 | 0.7736 | 0.9284 | **0.000439s** | **0.000000044s** |
| Decision Tree depth 2 | 0.3060 | 0.9318 | 0.4607 | 0.2854 | 0.000735s | 0.000000074s |
| Random Forest | 0.5513 | **0.9773** | 0.7049 | **0.9669** | 0.004302s | 0.000000432s |
| XGBoost | **0.9211** | 0.7955 | **0.8537** | 0.9399 | 0.001149s | 0.000000115s |

Current interpretation:

- **Random Forest**: best recall + best PR AUC, only `FN=1` in latest test run.
- **XGBoost**: best F1 and precision, but recall lower.
- **Logistic Regression**: fastest and strong simple baseline.

For fraud goal where missed fraud is expensive, current strongest candidate from the original comparison is:

```text
Random Forest
```

Latest documented tuned LightGBM run (`--tune-n-iter 100`) reached `recall=1.0000`, `roc_auc=0.9999`, and `f1=0.9286` with best params:

```text
{'subsample': 0.6, 'num_leaves': 127, 'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.01}
```

See [Step 7: Hyperparameter Tuning](docs/features/step-7-hyperparameter-tuning.md).

## Metrics

Current metrics logged:

- `training_accuracy`
- `test_accuracy`
- `precision`
- `recall`
- `f1`
- `pr_auc`
- `roc_auc`
- `val_threshold`
- `val_precision`
- `val_recall`
- `val_f1`
- `val_pr_auc`
- `val_roc_auc`
- `split_train`
- `split_val`
- `split_test`
- `predict_proba_latency_s`
- `predict_proba_latency_per_row_s`
- `single_row_latency_s`
- `inference_latency_s` — backward-compatible alias

## Artifacts

Runs are saved under:

```text
artifacts/runs/<timestamp>-<id>/
```

Each run can include:

```text
config.json
metrics.json
model.txt       # LightGBM
model.joblib    # sklearn / XGBoost
```

Security note:

- `model.joblib` uses pickle-based serialization.
- Only load joblib artifacts from trusted sources.
- See [Artifact Security](docs/artifact-security.md).

## Development

Run tests:

```bash
uv run pytest -q
```

Train one model:

```bash
uv run fraud-detect-train --data-path data/creditcard.csv --model random-forest
```

Train with validation-safe threshold tuning:

```bash
uv run fraud-detect-train \
  --data-path data/creditcard.csv \
  --batch-size 50000 \
  --imbalance-strategy scale-pos-weight \
  --val-size 0.1 \
  --threshold-objective target-recall \
  --target-recall 0.95 \
  --model random-forest
```

## Next Steps

Recommended next steps:

1. Add `score_transaction` / `predict_one` API.
2. Add safe model loading path with trust checks.
3. Persist tuning metadata as a dedicated artifact report.
4. Evaluate bigger batch or full dataset.
5. Add streaming-style inference test.
