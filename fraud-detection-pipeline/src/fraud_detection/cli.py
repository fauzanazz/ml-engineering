from argparse import ArgumentParser, ArgumentTypeError
from pathlib import Path

from fraud_detection.artifacts import make_run_dir, write_artifacts
from fraud_detection.models import LightGbmFactory
from fraud_detection.thresholds import sweep_thresholds, validate_threshold
from fraud_detection.training import train_one_batch


def test_size(value: str) -> float:
    size = float(value)
    if not 0 < size < 1:
        raise ArgumentTypeError("test_size must satisfy 0 < test_size < 1")
    return size


def decision_threshold(value: str) -> float:
    threshold = float(value)
    try:
        return validate_threshold(threshold)
    except ValueError as exc:
        raise ArgumentTypeError(str(exc)) from exc


def parse_args(argv: list[str] | None = None):
    parser = ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    import numpy as np

    from fraud_detection.thresholds import apply_threshold
    from fraud_detection.metrics import SklearnMetricsAdapter

    args = parse_args(argv)
    factory = LightGbmFactory()
    result = train_one_batch(
        data_path=args.data_path,
        model_factory=factory,
        batch_size=args.batch_size,
        target_column=args.target_column,
        test_size=args.test_size,
        imbalance_strategy=args.imbalance_strategy,
        decision_threshold=args.decision_threshold,
    )
    m = result.metrics
    print(f"training_accuracy={result.training_accuracy:.4f}")
    print(f"test_accuracy={result.test_accuracy:.4f}")
    print(f"precision={m.precision:.4f}")
    print(f"recall={m.recall:.4f}")
    print(f"f1={m.f1:.4f}")
    print(f"pr_auc={m.pr_auc:.4f}")

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
        run_dir = make_run_dir(args.artifact_dir)
        config = {
            "data_path": str(args.data_path),
            "batch_size": args.batch_size,
            "test_size": args.test_size,
            "imbalance_strategy": args.imbalance_strategy,
            "model_name": type(factory).__name__,
            "decision_threshold": args.decision_threshold,
        }
        write_artifacts(run_dir, result=result, config=config, model=result.model)
        print(f"artifacts saved to {run_dir}")
