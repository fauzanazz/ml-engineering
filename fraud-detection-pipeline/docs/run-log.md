# Historical Run Log

Manual record of notable runs before Step 4 artifact writing.

---

## Run 1 — Step 1 one-batch baseline (batch_size=256)

**Config**
- batch_size: 256
- imbalance_strategy: none
- model: LightGbmFactory

**Metrics**
- training_accuracy: 1.0000
- test_accuracy: (misleading — all predictions were 0)
- precision: 0.0000
- recall: 0.0000
- f1: 0.0000
- pr_auc: (not computed)

**Notes:** First rows of creditcard.csv contain no fraud. Batch too small, no positive samples in test set. Accuracy of 1.0 is deceptive — model predicts all-negative.

---

## Run 2 — Step 2 time-split baseline (batch_size=10000, strategy=none)

**Config**
- batch_size: 10000
- test_size: 0.2
- imbalance_strategy: none
- model: LightGbmFactory

**Metrics**
- training_accuracy: 1.0000
- test_accuracy: 0.9985
- precision: 0.8125
- recall: 1.0000
- f1: 0.8966
- pr_auc: 0.9682

**Notes:** Time-ordered split ensures test set contains later (more fraudulent) rows. Strong recall but training_accuracy=1.0 signals overfitting on train.

---

## Run 3 — Step 3 scale-pos-weight (batch_size=10000, strategy=scale-pos-weight)

**Config**
- batch_size: 10000
- test_size: 0.2
- imbalance_strategy: scale-pos-weight
- model: LightGbmFactory

**Metrics**
- training_accuracy: 0.9989
- test_accuracy: 0.9970
- precision: 0.7059
- recall: 0.9231
- f1: 0.8000
- pr_auc: 0.6928

**Notes:** `scale_pos_weight = neg/pos ≈ 578`. Training accuracy dropped from 1.0 — less overfit on negatives. Precision traded off vs recall. pr_auc lower than run 2, suggesting the weight amplification hurts score calibration on this small batch.

---

## Run 4 — Step 5 Threshold Tuning for Recall (batch_size=50000, strategy=scale-pos-weight, threshold=0.5)

**Perintah**
```
uv run fraud-detect-train --data-path "data/creditcard.csv" --batch-size 50000 --imbalance-strategy scale-pos-weight --decision-threshold 0.5 --threshold-sweep
```

**Metrik**
- training_accuracy: 0.9647
- test_accuracy: 0.9825
- precision: 0.2009
- recall: 1.0000
- f1: 0.3346
- pr_auc: 0.2750

**Threshold Sweep**

| Threshold | Precision | Recall | F1     | FP  | FN |
|-----------|-----------|--------|--------|-----|----|
| 0.05      | 0.0791    | 1.0000 | 0.1467 | 512 | 0  |
| 0.10      | 0.1796    | 1.0000 | 0.3045 | 201 | 0  |
| 0.20      | 0.2009    | 1.0000 | 0.3346 | 175 | 0  |
| 0.30      | 0.2009    | 1.0000 | 0.3346 | 175 | 0  |
| 0.50      | 0.2009    | 1.0000 | 0.3346 | 175 | 0  |
| 0.70      | 0.2009    | 1.0000 | 0.3346 | 175 | 0  |
| 0.80      | 0.2009    | 1.0000 | 0.3346 | 175 | 0  |
| 0.90      | 0.2009    | 1.0000 | 0.3346 | 175 | 0  |
| 0.95      | 0.2009    | 1.0000 | 0.3346 | 175 | 0  |

**Artefak**
`artifacts/runs/20260508T195212940609Z-8e6fa580/`

**Interpretasi**
Full sweep now covers lower (0.05, 0.10) and upper (0.70–0.95) thresholds. Thresholds 0.20–0.95 produce identical operating point (FP=175, FN=0) — model probabilities are saturated enough that raising the threshold past 0.20 doesn't change any classification. Lower thresholds 0.05/0.10 only add FP with no recall gain. Threshold 0.50 remains chosen for this batch.

**Lihat juga:** [docs/features/step-5-threshold-tuning-for-recall.md](features/step-5-threshold-tuning-for-recall.md)

---

## Run 5 — Step 6 Feature Engineering & Validation Split (batch_size=50000, strategy=scale-pos-weight, val_size=0.1, objective=target-recall)

**Perintah**
```
uv run fraud-detect-train --data-path "data/creditcard.csv" --batch-size 50000 --imbalance-strategy scale-pos-weight --val-size 0.1 --threshold-objective target-recall --target-recall 0.95
```

**Config**
- batch_size: 50000
- test_size: 0.2
- val_size: 0.1
- imbalance_strategy: scale-pos-weight
- model: LightGbmFactory
- threshold_objective: target-recall
- target_recall: 0.95
- val_threshold: 0.95

**Effective Split Counts**
- train: 34860
- val: 4961
- test: 9959

Counts are lower than raw split sizes because duplicate rows are removed per split and cross-split exact feature overlaps are removed to reduce leakage.

**Validation Metrics**
- val_precision: 0.0182
- val_recall: 1.0000
- val_f1: 0.0357
- val_pr_auc: 0.0294

**Test Metrics**
- training_accuracy: 0.9627
- test_accuracy: 0.9864
- precision: 0.2429
- recall: 0.9773
- f1: 0.3891
- pr_auc: 0.3390

**Artefak**
`artifacts/runs/20260509T075706256334Z-d5b2dd38/`

**Interpretasi**
Step 6 fixes Step 5 test-peeking by selecting the operating threshold on validation only, then applying that threshold once to test. The chosen validation threshold is `0.95`, meeting the target recall objective on validation (`val_recall=1.0`). Test recall stays high (`0.9773`) and precision improves versus Run 4 (`0.2429` vs `0.2009`), with better f1 (`0.3891` vs `0.3346`) and pr_auc (`0.3390` vs `0.2750`). Validation pr_auc is very low (`0.0294`), so the validation slice may be noisy/small or score calibration remains weak.

**Note on Step 5**
Run 4 (Step 5) selected threshold by sweeping over test scores — test-peeking. Those metrics are exploratory only. Step 6 corrects this: threshold selection now uses validation set exclusively; test remains held-out.

**Tests**
164 passing (`uv run pytest -q`).

**See:** [docs/features/step-6-feature-engineering-validation-split.md](features/step-6-feature-engineering-validation-split.md)

---

## Run 7 — Step 7: Optuna TPE Hyperparameter Tuning, 500 Candidates

**Command pattern**

```bash
uv run fraud-detect-train --data-path data/creditcard.csv --batch-size 10000 \
  --model <model> --tune --tune-n-candidates 500 --no-artifacts
```

**Models tuned**

- `lightgbm`
- `random-forest`
- `xgboost`

**Results**

| Model | Tuning CV ROC AUC | Precision | Recall | F1 | PR AUC | ROC AUC | Single-row latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Random Forest** | 0.9990 | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | 0.003094500s |
| LightGBM | 0.9997 | 0.8667 | **1.0000** | 0.9286 | 0.9897 | 0.9999 | **0.000319167s** |
| XGBoost | **0.9998** | 0.8571 | 0.9231 | 0.8889 | 0.9757 | 0.9998 | 0.000534916s |

**Best model by held-out test metrics**

```text
Random Forest
```

Best params:

```text
{'n_estimators': 200, 'max_depth': 5, 'min_samples_split': 2, 'min_samples_leaf': 4, 'max_features': 'sqrt'}
```

**Best latency model**

```text
LightGBM
```

LightGBM keeps perfect recall (`1.0000`) with much lower single-row latency (`0.000319167s`) than Random Forest (`0.003094500s`).

**Interpretasi**

Random Forest is the current winner for this `batch_size=10000` research batch because it reaches perfect precision, recall, F1, PR AUC, and ROC AUC on held-out test. The result is promising but should be verified on a bigger batch or full dataset because perfect scores on a small fraud slice can be unstable.

**See:** [docs/features/step-7-hyperparameter-tuning.md](features/step-7-hyperparameter-tuning.md)
