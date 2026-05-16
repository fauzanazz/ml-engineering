"""Batch recommendations from JSONL input using active registry model."""

from pathlib import Path
import argparse
import json
from time import perf_counter
from typing import Any

from .predict import recommend_top_k_from_registry
from .train import get_active_model

DEFAULT_MODEL_NAME = "movielens-popularity"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                rows.append({"request_id": f"line-{line_number}", "_row_error": f"invalid JSON: {error.msg}"})
                continue
            if not isinstance(row, dict):
                rows.append({"request_id": f"line-{line_number}", "_row_error": "row must be a JSON object"})
                continue
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")


def _request_id(row: dict[str, Any], fallback: str) -> str:
    request_id = row.get("request_id")
    if request_id is None or str(request_id).strip() == "":
        return fallback
    return str(request_id)


def _top_k(row: dict[str, Any]) -> int:
    top_k = int(row.get("top_k", 10))
    if top_k < 1:
        raise ValueError(f"top_k must be positive, got {top_k}")
    return top_k


def _user_id(row: dict[str, Any]) -> int | None:
    user_id = row.get("user_id")
    if user_id is None:
        return None
    return int(user_id)


def calculate_batch_performance(
    input_path: Path,
    output_path: Path,
    input_row_count: int,
    success_row_count: int,
    error_row_count: int,
    duration_seconds: float,
) -> dict[str, object]:
    throughput = input_row_count / duration_seconds if input_row_count > 0 and duration_seconds > 0 else 0.0
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "input_row_count": input_row_count,
        "success_row_count": success_row_count,
        "error_row_count": error_row_count,
        "duration_seconds": round(duration_seconds, 6),
        "throughput_rows_per_second": round(throughput, 6),
        "status": _batch_status(success_row_count, error_row_count),
    }


def run_batch_recommendations(
    registry_path: Path,
    input_path: Path,
    output_path: Path,
    model_name: str = DEFAULT_MODEL_NAME,
    report_path: Path | None = None,
) -> dict[str, object]:
    active_model = get_active_model(registry_path, model_name)
    if active_model is None:
        raise ValueError(f"active model not found: {model_name}")

    started_at = perf_counter()
    output_rows = []
    succeeded = 0
    failed = 0
    input_rows = _read_jsonl(input_path)
    model_version = str(active_model["version"])
    active_model_name = str(active_model["model_name"])

    for index, row in enumerate(input_rows, start=1):
        request_id = _request_id(row, f"row-{index}")
        try:
            if "_row_error" in row:
                raise ValueError(str(row["_row_error"]))
            recommendations = recommend_top_k_from_registry(registry_path, model_name, _top_k(row), _user_id(row))
        except Exception as error:
            failed += 1
            output_rows.append(
                {
                    "request_id": request_id,
                    "model_name": active_model_name,
                    "version": model_version,
                    "recommendations": [],
                    "error": str(error),
                }
            )
            continue

        succeeded += 1
        output_rows.append(
            {
                "request_id": request_id,
                "model_name": active_model_name,
                "version": model_version,
                "recommendations": recommendations,
                "error": None,
            }
        )

    _write_jsonl(output_path, output_rows)
    duration_seconds = perf_counter() - started_at
    performance_report = calculate_batch_performance(
        input_path=input_path,
        output_path=output_path,
        input_row_count=len(input_rows),
        success_row_count=succeeded,
        error_row_count=failed,
        duration_seconds=duration_seconds,
    )
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(performance_report, indent=2, sort_keys=True) + "\n")
    return {
        "input_rows": len(input_rows),
        "succeeded": succeeded,
        "failed": failed,
        "output_path": str(output_path),
    }


def _batch_status(success_row_count: int, error_row_count: int) -> str:
    if success_row_count == 0:
        return "failed"
    if error_row_count > 0:
        return "degraded"
    return "passed"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run JSONL batch recommendations using active registry model.")
    parser.add_argument("--registry-path", type=Path, required=True)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    args = parser.parse_args()

    summary = run_batch_recommendations(
        args.registry_path,
        args.input_path,
        args.output_path,
        args.model_name,
        report_path=args.report_path,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
