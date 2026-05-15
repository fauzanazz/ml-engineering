# Artifacts

Training runs write timestamped directories under `artifacts/`.

Files:

- `config.json`: CLI and model settings.
- `metrics.json`: validation metrics.
- `model.joblib`: trained estimator.
- `feature_pipeline.joblib`: fitted preprocessing pipeline.
- `submission.csv`: Kaggle-ready predictions when test data exists.

Only load `joblib` artifacts from trusted runs.
