# Dataset Source

This repository does **not** commit the raw dataset because `data/creditcard.csv` is large and should stay local.

## Source

Dataset used in this project:

- **Name:** Credit Card Fraud Detection
- **Provider:** Machine Learning Group - ULB
- **Source:** Kaggle
- **URL:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **Local path expected by examples:** `data/creditcard.csv`

## How to set it up

1. Download the dataset from Kaggle:

   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

2. Extract `creditcard.csv`.

3. Put it here:

   ```text
   fraud-detection-pipeline/data/creditcard.csv
   ```

4. Run a smoke training command:

   ```bash
   uv run fraud-detect-train --data-path data/creditcard.csv --batch-size 10000 --model lightgbm --no-artifacts
   ```

## Dataset notes

The dataset contains anonymized European credit-card transactions from September 2013. Features `V1`–`V28` are PCA-transformed for privacy, with raw `Time`, `Amount`, and target label `Class`.

Label meanings:

- `Class=0`: legitimate transaction
- `Class=1`: fraud transaction

The dataset is highly imbalanced, so this project reports fraud-focused metrics such as PR AUC, ROC AUC, recall, precision, F1, and threshold-tuned validation metrics.

## Git policy

`data/creditcard.csv` should remain untracked. Keep raw data local and commit only code, docs, tests, and lightweight artifacts.
