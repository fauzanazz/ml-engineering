from pathlib import Path
import json

from fastapi.testclient import TestClient

from ml_production_ecosystem.recommendation.api import create_app
from ml_production_ecosystem.recommendation.train import register_model_version, train_popularity_recommender

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


def test_health_returns_ok(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path / "registry" / "models.json"))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_models_active_returns_active_model_metadata(tmp_path: Path) -> None:
    registry_path = _registry_with_active_model(tmp_path)
    client = TestClient(create_app(registry_path))

    response = client.get("/models/active")

    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "movielens-popularity"
    assert body["version"] == "api-v1"
    assert body["stage"] == "production"
    assert "artifact_uri" in body
    assert "metrics_uri" in body


def test_predict_v1_returns_recommendations_from_active_model(tmp_path: Path) -> None:
    registry_path = _registry_with_active_model(tmp_path)
    client = TestClient(create_app(registry_path))

    response = client.post("/predict/v1", json={"user_id": 10, "top_k": 2})

    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "movielens-popularity"
    assert body["version"] == "api-v1"
    assert body["request_id"]
    assert [item["movieId"] for item in body["recommendations"]] == [1, 3]


def test_metrics_json_tracks_successful_prediction(tmp_path: Path) -> None:
    registry_path = _registry_with_active_model(tmp_path)
    client = TestClient(create_app(registry_path))

    initial_metrics = client.get("/metrics.json").json()
    response = client.post("/predict/v1", json={"user_id": 10, "top_k": 2})
    updated_metrics = client.get("/metrics.json").json()

    assert response.status_code == 200
    assert initial_metrics == {
        "prediction_request_count": 0,
        "prediction_error_count": 0,
        "prediction_latency_ms_avg": 0.0,
        "prediction_latency_ms_last": 0.0,
        "last_model_name": None,
        "last_model_version": None,
    }
    assert updated_metrics["prediction_request_count"] == 1
    assert updated_metrics["prediction_error_count"] == 0
    assert updated_metrics["prediction_latency_ms_avg"] > 0
    assert updated_metrics["prediction_latency_ms_last"] > 0
    assert updated_metrics["last_model_name"] == "movielens-popularity"
    assert updated_metrics["last_model_version"] == "api-v1"


def test_metrics_returns_prometheus_text_for_successful_prediction(tmp_path: Path) -> None:
    registry_path = _registry_with_active_model(tmp_path)
    client = TestClient(create_app(registry_path))

    response = client.post("/predict/v1", json={"user_id": 10, "top_k": 2})
    metrics_response = client.get("/metrics")

    assert response.status_code == 200
    assert metrics_response.status_code == 200
    assert metrics_response.headers["content-type"].startswith("text/plain")
    labels = '{model_name="movielens-popularity",model_version="api-v1"}'
    metrics_text = metrics_response.text
    assert f"foundation_prediction_requests_total{labels} 1" in metrics_text
    assert f"foundation_prediction_errors_total{labels} 0" in metrics_text
    assert f"foundation_prediction_latency_ms_sum{labels} " in metrics_text
    assert f"foundation_prediction_latency_ms_count{labels} 1" in metrics_text
    assert f"foundation_prediction_latency_ms_last{labels} " in metrics_text


def test_predict_v1_writes_success_prediction_log(tmp_path: Path) -> None:
    registry_path = _registry_with_active_model(tmp_path)
    prediction_log_path = tmp_path / "logs" / "predictions.jsonl"
    client = TestClient(create_app(registry_path, prediction_log_path=prediction_log_path))

    response = client.post("/predict/v1", json={"user_id": 10, "top_k": 2})

    body = response.json()
    rows = [json.loads(line) for line in prediction_log_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["request_id"] == body["request_id"]
    assert rows[0]["user_id"] == 10
    assert rows[0]["top_k"] == 2
    assert rows[0]["model_name"] == "movielens-popularity"
    assert rows[0]["model_version"] == "api-v1"
    assert rows[0]["recommendation_movie_ids"] == [1, 3]
    assert rows[0]["status"] == "success"
    assert rows[0]["latency_ms"] > 0
    assert rows[0]["error"] is None
    assert "requested_at" in rows[0]


def test_drift_returns_baseline_comparison(tmp_path: Path) -> None:
    registry_path = _registry_with_active_model(tmp_path)
    prediction_log_path = tmp_path / "logs" / "predictions.jsonl"
    client = TestClient(create_app(registry_path, prediction_log_path=prediction_log_path))
    client.post("/predict/v1", json={"user_id": 10, "top_k": 2})

    response = client.get("/drift")

    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "movielens-popularity"
    assert body["version"] == "api-v1"
    assert body["sample_size"] == 2
    assert body["baseline_size"] == 10
    assert body["drift_score"] == 0.0


def test_predict_v1_returns_404_when_active_model_missing(tmp_path: Path) -> None:
    prediction_log_path = tmp_path / "logs" / "predictions.jsonl"
    client = TestClient(create_app(tmp_path / "registry" / "models.json", prediction_log_path=prediction_log_path))

    response = client.post("/predict/v1", json={"top_k": 2})
    metrics = client.get("/metrics.json").json()
    metrics_text = client.get("/metrics").text
    rows = [json.loads(line) for line in prediction_log_path.read_text().splitlines()]

    assert response.status_code == 404
    assert response.json() == {"detail": "active model not found: movielens-popularity"}
    assert metrics["prediction_request_count"] == 1
    assert metrics["prediction_error_count"] == 1
    assert metrics["prediction_latency_ms_last"] > 0
    assert 'foundation_prediction_errors_total{model_name="unknown",model_version="unknown"} 1' in metrics_text
    assert len(rows) == 1
    assert rows[0]["request_id"]
    assert rows[0]["status"] == "error"
    assert rows[0]["error"] == "active model not found: movielens-popularity"
    assert rows[0]["recommendation_movie_ids"] == []
