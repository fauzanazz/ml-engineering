from pathlib import Path

from fastapi.testclient import TestClient

from recommendation.api import create_app
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
    assert [item["movieId"] for item in body["recommendations"]] == [1, 3]


def test_predict_v1_returns_404_when_active_model_missing(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path / "registry" / "models.json"))

    response = client.post("/predict/v1", json={"top_k": 2})

    assert response.status_code == 404
    assert response.json() == {"detail": "active model not found: movielens-popularity"}
