# {{project_name}}

Kaggle competition scaffold focused on fast local iteration.

## Commands

```bash
uv run python -m {{package_name}}.train
uv run python -m {{package_name}}.submission
```

## Flow

1. Put competition files in `data/raw`.
2. Build features in `{{package_name}}/features.py`.
3. Train baseline in `{{package_name}}/train.py`.
4. Create `submission.csv` with `{{package_name}}/submission.py`.
