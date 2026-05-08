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
