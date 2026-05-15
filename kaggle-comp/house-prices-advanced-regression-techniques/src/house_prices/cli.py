import argparse
import shutil
from pathlib import Path

from house_prices.artifacts import make_run_dir, write_artifacts
from house_prices.data import load_kaggle_test_data
from house_prices.models import make_model_factory
from house_prices.training import TrainingResult, predict_submission, train_model

MODEL_CHOICES = ["lightgbm", "random-forest", "xgboost", "ensemble"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train house-prices regression pipeline")
    parser.add_argument("--data-path", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--test-path", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--target-column", default="SalePrice")
    parser.add_argument("--id-column", default="Id")
    parser.add_argument("--model", choices=[*MODEL_CHOICES, "all"], default="lightgbm")
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--run-id")
    return parser


def _model_names(selected_model: str) -> list[str]:
    if selected_model == "all":
        return MODEL_CHOICES
    return [selected_model]


def _config(args: argparse.Namespace, model_name: str, result: TrainingResult) -> dict[str, object]:
    return {
        "data_path": str(args.data_path),
        "test_path": str(args.test_path),
        "target_column": args.target_column,
        "id_column": args.id_column,
        "model": model_name,
        "val_size": args.val_size,
        "random_state": args.random_state,
        "split_strategy": result.split_strategy,
        "target_transform": result.target_transform,
        "selection_metric": "validation_log_rmse",
        "leakage_policy": "train-only validation; no Kaggle test labels or leaderboard feedback",
    }


def _write_one_run(args: argparse.Namespace, model_name: str, test_features) -> tuple[Path, TrainingResult]:
    factory = make_model_factory(model_name, random_state=args.random_state)
    result = train_model(
        args.data_path,
        factory,
        target_column=args.target_column,
        val_size=args.val_size,
        random_state=args.random_state,
    )
    submission = None
    if test_features is not None:
        submission = predict_submission(
            result.model,
            result.feature_pipeline,
            test_features,
            id_column=args.id_column,
        )

    run_id = f"{args.run_id}-{model_name}" if args.run_id and len(_model_names(args.model)) > 1 else args.run_id
    run_dir = write_artifacts(
        make_run_dir(args.artifacts_dir, run_id),
        result=result,
        config=_config(args, model_name, result),
        submission=submission,
    )
    return run_dir, result


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    test_features = load_kaggle_test_data(args.test_path) if args.test_path.exists() else None
    runs = [_write_one_run(args, model_name, test_features) for model_name in _model_names(args.model)]
    best_run_dir, best_result = min(runs, key=lambda run: run[1].metrics.log_rmse)

    if (best_run_dir / "submission.csv").exists():
        shutil.copyfile(best_run_dir / "submission.csv", args.artifacts_dir / "best_submission.csv")

    print(f"best_run_dir={best_run_dir}")
    print(f"best_log_rmse={best_result.metrics.log_rmse:.6f}")
    for run_dir, result in runs:
        print(f"run_dir={run_dir} log_rmse={result.metrics.log_rmse:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
