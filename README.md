# ML Engineering

Production-oriented ML engineering workspace. Isi repo bukan cuma LightGBM fraud detection lagi; target repo ini jadi kumpulan project end-to-end untuk pipeline training, evaluation, artifact logging, model comparison, dan production readiness.

## Projects

| Project | Description | Status |
|---------|-------------|--------|
| [fraud-detection-pipeline](./fraud-detection-pipeline/) | Fraud detection pipeline dengan chronological split, leakage checks, feature engineering, imbalance handling, threshold tuning, artifact logging, latency metric, dan multi-model tuning (LightGBM, Random Forest, XGBoost) | Active |
| [ml-production-ecosystem](./ml-production-ecosystem/) | Workspace untuk perluasan ekosistem ML production di repo ini | Planned |

## Focus Areas

- Reproducible train / validation / test pipelines
- Data leakage prevention
- Imbalanced classification evaluation
- Model comparison and hyperparameter tuning
- Run artifacts and experiment tracking foundations
- Production-oriented metrics, including latency

## Repository Structure

```text
.
├── fraud-detection-pipeline/
├── ml-production-ecosystem/
└── README.md
```

## License

Personal use.
