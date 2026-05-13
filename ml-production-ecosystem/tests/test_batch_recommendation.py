from pathlib import Path
import json
import subprocess
import sys

from recommendation.batch import run_batch_recommendations
from recommendation.train import register_model_version, train_popularity_recommender

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "recommendation"


def _registry_with_active_model(tmp_path: Path) -> Path:
    result = train_popularity_recommender(
        ratings_path=FIXTURE_DIR / "ratings.csv",
        movies_path=FIXTURE_DIR / "movies.csv",
        artifact_dir=tmp_path / "artifacts",
        version="api-v1",
        min_rating=4.0,
    )
    registry_path = tmp_path / "registry" / "models.json"
    register_model_version(
        registry_path=registry_path,
        model_name="movielens-popularity",
        version="api-v1",
        artifact_uri=result.uri,
        metrics_uri=result.metrics_uri,
        stage="production",
        set_active=True,
    )
    return registry_path


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_batch_recommendations_write_success_rows(tmp_path: Path) -> None:
    registry_path = _registry_with_active_model(tmp_path)
    input_path = tmp_path / "batch-input.jsonl"
    output_path = tmp_path / "batch-output.jsonl"
    input_path.write_text(json.dumps({"request_id": "batch-1", "user_id": 10, "top_k": 2}) + "\n")

    summary = run_batch_recommendations(registry_path, input_path, output_path)

    assert summary == {
        "input_rows": 1,
        "succeeded": 1,
        "failed": 0,
        "output_path": str(output_path),
    }
    rows = _read_jsonl(output_path)
    assert rows[0]["request_id"] == "batch-1"
    assert rows[0]["model_name"] == "movielens-popularity"
    assert rows[0]["version"] == "api-v1"
    assert [item["movieId"] for item in rows[0]["recommendations"]] == [1, 3]
    assert rows[0]["error"] is None


def test_batch_recommendations_write_error_rows_and_continue(tmp_path: Path) -> None:
    registry_path = _registry_with_active_model(tmp_path)
    input_path = tmp_path / "batch-input.jsonl"
    output_path = tmp_path / "batch-output.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps({"request_id": "bad-1", "user_id": 10, "top_k": 0}),
                json.dumps({"request_id": "batch-2", "user_id": 10, "top_k": 1}),
            ]
        )
        + "\n"
    )

    summary = run_batch_recommendations(registry_path, input_path, output_path)

    assert summary == {
        "input_rows": 2,
        "succeeded": 1,
        "failed": 1,
        "output_path": str(output_path),
    }
    rows = _read_jsonl(output_path)
    assert rows[0] == {
        "request_id": "bad-1",
        "model_name": "movielens-popularity",
        "version": "api-v1",
        "recommendations": [],
        "error": "top_k must be positive, got 0",
    }
    assert rows[1]["request_id"] == "batch-2"
    assert [item["movieId"] for item in rows[1]["recommendations"]] == [1]
    assert rows[1]["error"] is None


def test_batch_recommendation_cli_prints_summary(tmp_path: Path) -> None:
    registry_path = _registry_with_active_model(tmp_path)
    input_path = tmp_path / "batch-input.jsonl"
    output_path = tmp_path / "batch-output.jsonl"
    input_path.write_text(json.dumps({"request_id": "batch-cli", "user_id": 10, "top_k": 1}) + "\n")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "recommendation.batch",
            "--registry-path",
            str(registry_path),
            "--input-path",
            str(input_path),
            "--output-path",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "input_rows": 1,
        "succeeded": 1,
        "failed": 0,
        "output_path": str(output_path),
    }
    assert _read_jsonl(output_path)[0]["request_id"] == "batch-cli"
