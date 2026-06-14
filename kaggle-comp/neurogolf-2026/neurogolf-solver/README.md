# neurogolf-solver

Kaggle competition scaffold focused on fast local iteration.

## Commands

```bash
uv run python -m neurogolf_solver.train
uv run python -m neurogolf_solver.submission
```

## Flow

1. Put competition files in `data/raw`.
2. Build features in `neurogolf_solver/features.py`.
3. Train baseline in `neurogolf_solver/train.py`.
4. Create `submission.csv` with `neurogolf_solver/submission.py`.
