import math
import sys
import time
from argparse import ArgumentParser, ArgumentTypeError
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from fraud_detection.artifacts import (
    SCHEMA_VERSION,
    collect_dataset_metadata,
    collect_runtime_metadata,
    make_run_dir,
    write_artifacts,
)
from fraud_detection.models import FACTORY_MAP
from fraud_detection.thresholds import sweep_thresholds, validate_threshold
from fraud_detection.training import train_one_batch


def _fraction(name: str, value: str) -> float:
    size = float(value)
    if not 0 < size < 1:
        raise ArgumentTypeError(f"{name} must satisfy 0 < {name} < 1")
    return size


def test_size(value: str) -> float:
    return _fraction("test_size", value)


def val_size(value: str) -> float:
    return _fraction("val_size", value)


def decision_threshold(value: str) -> float:
    threshold = float(value)
    try:
        return validate_threshold(threshold)
    except ValueError as exc:
        raise ArgumentTypeError(str(exc)) from exc


def target_recall(value: str) -> float:
    recall = float(value)
    if not 0.0 <= recall <= 1.0:
        raise ArgumentTypeError(
            f"target_recall must satisfy 0 <= target_recall <= 1, got {recall}"
        )
    return recall


def tune_cv(value: str) -> int:
    folds = int(value)
    if not 2 <= folds <= 10:
        raise ArgumentTypeError("tune_cv must satisfy 2 <= tune_cv <= 10")
    return folds


def parse_args(argv: list[str] | None = None):
    parser = ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-column", default="Class")
    parser.add_argument("--test-size", type=test_size, default=0.2)
    parser.add_argument(
        "--imbalance-strategy",
        choices=["none", "scale-pos-weight"],
        default="none",
        help="none: no adjustment; scale-pos-weight: weight positives by neg/pos ratio",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/runs"),
        help="base directory for run artifacts (default: artifacts/runs)",
    )
    parser.add_argument(
        "--no-artifacts",
        action="store_true",
        default=False,
        help="skip writing artifacts",
    )
    parser.add_argument(
        "--decision-threshold",
        type=decision_threshold,
        default=0.5,
        help="probability cutoff for fraud label (default: 0.5)",
    )
    parser.add_argument(
        "--threshold-sweep",
        action="store_true",
        default=False,
        help="print precision/recall table across [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95]",
    )
    parser.add_argument(
        "--val-size",
        type=val_size,
        default=None,
        help="fraction of data for validation set; enables three-way split and threshold selection",
    )
    parser.add_argument(
        "--threshold-objective",
        choices=["f1", "target-recall"],
        default="f1",
        help="objective for threshold selection on validation set (default: f1)",
    )
    parser.add_argument(
        "--target-recall",
        type=target_recall,
        default=None,
        help="minimum recall floor for target-recall objective (must be in [0, 1])",
    )
    parser.add_argument(
        "--model",
        choices=list(FACTORY_MAP),
        default="lightgbm",
        help="model to train (default: lightgbm)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        default=False,
        help="run adaptive hyperparameter search before training (lightgbm, random-forest, xgboost)",
    )
    parser.add_argument(
        "--tune-n-candidates",
        type=int,
        default=10,
        dest="tune_n_candidates",
        help=(
            "number of Optuna TPE trials (default: 10, max: 500); each trial uses "
            "forward TimeSeriesSplit CV on chronological training data with fold-local "
            "preprocessing and average precision"
        ),
    )
    parser.add_argument(
        "--tune-cv",
        type=tune_cv,
        default=3,
        help="number of forward TimeSeriesSplit folds (default: 3, range: 2..10)",
    )
    parsed = parser.parse_args(argv)
    if parsed.target_recall is not None and parsed.threshold_objective != "target-recall":
        parser.error("--target-recall requires --threshold-objective target-recall")
    if parsed.val_size is None and parsed.threshold_objective != "f1":
        parser.error("--val-size is required when --threshold-objective is not f1")
    if parsed.val_size is None and parsed.target_recall is not None:
        parser.error("--val-size is required when --target-recall is provided")
    if parsed.val_size is not None and parsed.decision_threshold != 0.5:
        parser.error(
            "--decision-threshold cannot be set when --val-size is provided; "
            "threshold is selected automatically on the validation set"
        )
    tunable_models = {"lightgbm", "random-forest", "xgboost"}
    if parsed.tune and parsed.model not in tunable_models:
        parser.error(f"--tune only supports {sorted(tunable_models)}, got '{parsed.model}'")
    if parsed.tune_n_candidates < 1:
        parser.error(
            f"--tune-n-candidates must be >= 1, got {parsed.tune_n_candidates}"
        )
    if parsed.tune_n_candidates > 500:
        parser.error(
            f"--tune-n-candidates must be <= 500, got {parsed.tune_n_candidates}"
        )
    return parsed


def main(argv: list[str] | None = None) -> None:
    started = time.perf_counter()
    started_at_utc = datetime.now(timezone.utc).isoformat()
    effective_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parse_args(effective_argv)
    dataset_metadata = (
        collect_dataset_metadata(args.data_path, target_column=args.target_column)
        if not args.no_artifacts
        else None
    )
    runtime_metadata = collect_runtime_metadata() if not args.no_artifacts else None
    tuning_result = None

    if args.tune:
        from fraud_detection.data import load_three_way_split, load_time_split_batch
        from fraud_detection.tuning import (
            tune_lightgbm,
            tune_random_forest,
            tune_xgboost,
        )

        if args.val_size is not None:
            train_features, _, _, train_target, _, _ = load_three_way_split(
                args.data_path,
                val_size=args.val_size,
                test_size=args.test_size,
                target_column=args.target_column,
                batch_size=args.batch_size,
            )
        else:
            train_features, _, train_target, _ = load_time_split_batch(
                args.data_path,
                args.batch_size,
                args.target_column,
                args.test_size,
            )
        tune_fn = {
            "lightgbm": tune_lightgbm,
            "random-forest": tune_random_forest,
            "xgboost": tune_xgboost,
        }[args.model]
        try:
            tuning_result = tune_fn(
                train_features,
                train_target,
                n_iter=args.tune_n_candidates,
                cv=args.tune_cv,
                random_state=args.seed,
                imbalance_strategy=args.imbalance_strategy,
            )
        except ValueError as exc:
            raise SystemExit(f"error: tuning failed — {exc}") from exc
        print(f"tuning_best_score={tuning_result.best_score:.4f}")
        print(f"tuning_best_params={tuning_result.best_params}")
        factory = tuning_result.best_factory
    else:
        factory = FACTORY_MAP[args.model](random_state=args.seed)

    result = train_one_batch(
        data_path=args.data_path,
        model_factory=factory,
        batch_size=args.batch_size,
        target_column=args.target_column,
        test_size=args.test_size,
        imbalance_strategy=args.imbalance_strategy,
        decision_threshold=args.decision_threshold,
        val_size=args.val_size,
        threshold_objective=args.threshold_objective,
        target_recall=args.target_recall,
    )
    duration_s = time.perf_counter() - started
    metrics = result.metrics
    print(f"training_accuracy={result.training_accuracy:.4f}")
    print(f"test_accuracy={result.test_accuracy:.4f}")
    print(f"precision={metrics.precision:.4f}")
    print(f"recall={metrics.recall:.4f}")
    print(f"f1={metrics.f1:.4f}")
    roc_auc = (
        "nan (undefined: single-class labels)"
        if math.isnan(metrics.roc_auc)
        else f"{metrics.roc_auc:.4f}"
    )
    print(f"pr_auc={metrics.pr_auc:.4f} roc_auc={roc_auc}")
    if result.predict_proba_latency_s is not None:
        print(f"predict_proba_latency_s={result.predict_proba_latency_s:.6f}")
    if result.predict_proba_latency_per_row_s is not None:
        print(
            f"predict_proba_latency_per_row_s={result.predict_proba_latency_per_row_s:.9f}"
        )
    if result.single_row_latency_s is not None:
        print(f"single_row_latency_s={result.single_row_latency_s:.9f}")

    if args.threshold_sweep:
        rows = sweep_thresholds(result.test_labels, result.test_scores)
        print(
            f"\n{'threshold':>10} {'precision':>10} {'recall':>10} {'f1':>8} {'FP':>6} {'FN':>6}"
        )
        for row in rows:
            print(
                f"{row.threshold:>10.2f} {row.precision:>10.4f} {row.recall:>10.4f}"
                f" {row.f1:>8.4f} {row.false_positives:>6} {row.false_negatives:>6}"
            )

    if not args.no_artifacts:
        tuning_config = {"enabled": tuning_result is not None}
        if tuning_result is not None:
            tuning_config.update(
                {
                    "n_trials": args.tune_n_candidates,
                    "scoring": tuning_result.scoring,
                    "cv_strategy": tuning_result.cv_strategy,
                    "cv_splits": tuning_result.cv_splits,
                    "best_score": tuning_result.best_score,
                    "best_params": tuning_result.best_params,
                }
            )
        model_params = (
            result.model.get_params(deep=False)
            if hasattr(result.model, "get_params")
            else {}
        )
        config = {
            "schema_version": SCHEMA_VERSION,
            "command": ["fraud-detect-train", *effective_argv],
            "dataset": dataset_metadata,
            "run": {
                "started_at_utc": started_at_utc,
                "duration_s": duration_s,
                "seed": args.seed,
            },
            "runtime": runtime_metadata,
            "training": {
                "model_key": args.model,
                "estimator_class": type(result.model).__name__,
                "model_params": model_params,
                "batch_size": args.batch_size,
                "imbalance_strategy": args.imbalance_strategy,
            },
            "split": {
                "val_size": args.val_size,
                "test_size": args.test_size,
                "audit": asdict(result.split_audit),
            },
            "tuning": tuning_config,
            "threshold": {
                "objective": args.threshold_objective if args.val_size is not None else "fixed",
                "target_recall": args.target_recall,
                "selected": result.effective_threshold,
                "target_met": result.threshold_target_met,
                "fallback_used": result.threshold_fallback_used,
            },
        }
        run_dir = make_run_dir(args.artifact_dir)
        write_artifacts(run_dir, result=result, config=config)
        print(f"artifacts saved to {run_dir}")
