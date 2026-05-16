from pathlib import Path

import pytest

from ml_production_ecosystem.recommendation.prediction_adapter import RecommenderPredictionAdapter
from ml_production_ecosystem.recommendation.train import register_model_version, train_popularity_recommender
from ml_production_ecosystem.shared.model_contracts import PredictionPort, PredictionRequest

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "recommendation"

def _registry_with_active_model(tmp_path: Path) -> Path:
    result = train_popularity_recommender(
        ratings_path=FIXTURE_DIR / "ratings.csv",
        movies_path=FIXTURE_DIR / "movies.csv",
        artifact_dir=tmp_path / "artifacts",
        version="adapter-v1",
    )
    registry_path = tmp_path / "registry" / "models.json"
    register_model_version(
        registry_path=registry_path,
        model_name=result.model_name,
        version=result.version,
        artifact_uri=result.uri,
        metrics_uri=result.metrics_uri,
        set_active=True,
    )
    return registry_path

def test_recommender_prediction_adapter_implements_generic_prediction_port(tmp_path: Path) -> None:
    adapter: PredictionPort = RecommenderPredictionAdapter(_registry_with_active_model(tmp_path))

    response = adapter.predict(
        PredictionRequest(
            records=[{"user_id": 1, "top_k": 2}, {"top_k": 1}],
            request_id="req-adapter",
        )
    )

    assert response.request_id == "req-adapter"
    assert response.model.name == "movielens-popularity"
    assert response.model.version == "adapter-v1"
    assert response.model.framework == "local-json"
    assert response.model.io_contract.prediction_key == "recommendations"
    assert len(response.predictions) == 2
    assert len(response.predictions[0]) == 2
    assert len(response.predictions[1]) == 1
    assert response.metadata["adapter"] == "ml_production_ecosystem.recommendation.prediction_adapter"

def test_recommender_prediction_adapter_fails_without_active_model(tmp_path: Path) -> None:
    adapter = RecommenderPredictionAdapter(tmp_path / "missing-registry.json")

    with pytest.raises(ValueError, match="active model not found"):
        adapter.predict(PredictionRequest(records=[{"top_k": 1}]))
