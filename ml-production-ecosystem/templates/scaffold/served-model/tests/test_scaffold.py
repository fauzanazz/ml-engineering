from fastapi.testclient import TestClient
import pytest

from {{package_name}}.api import Settings, app, create_app
from {{package_name}}.predict import predict


@pytest.mark.parametrize(
    ("features", "threshold", "expected"),
    [
        ({"a": 5.0}, 5.0, True),
        ({"a": 4.99}, 5.0, False),
        ({}, 0.0, True),
        ({"a": -1.0, "b": 0.5}, 0.0, False),
    ],
)
def test_predict_applies_threshold_boundary(features: dict[str, float], threshold: float, expected: bool) -> None:
    assert predict(features, threshold=threshold) is expected


def test_module_level_app_serves_health_endpoint() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_reports_model_identity_from_injected_settings() -> None:
    settings = Settings(model_name="churn-model", model_version="4.2.0")
    client = TestClient(create_app(settings))

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_name"] == "churn-model"
    assert body["model_version"] == "4.2.0"


def test_predict_endpoint_returns_boolean_prediction_matching_threshold() -> None:
    settings = Settings(model_name="churn-model", model_version="4.2.0", threshold=1.0)
    client = TestClient(create_app(settings))

    response = client.post("/predict", json={"features": {"a": 0.4, "b": 0.4}})

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] is False  # 0.4 + 0.4 < threshold 1.0
    assert body["model_name"] == "churn-model"
    assert body["model_version"] == "4.2.0"


def test_predict_and_health_report_identical_model_metadata() -> None:
    settings = Settings(model_name="churn-model", model_version="9.9.9")
    client = TestClient(create_app(settings))

    health_body = client.get("/health").json()
    predict_body = client.post("/predict", json={"features": {"a": 10.0}}).json()

    assert predict_body["model_name"] == health_body["model_name"]
    assert predict_body["model_version"] == health_body["model_version"]


def test_predict_rejects_missing_features_field() -> None:
    client = TestClient(create_app(Settings()))

    response = client.post("/predict", json={})

    assert response.status_code == 422


def test_predict_rejects_non_numeric_feature_value() -> None:
    client = TestClient(create_app(Settings()))

    response = client.post("/predict", json={"features": {"a": "not-a-number"}})

    assert response.status_code == 422


def test_settings_from_env_overrides_apply(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_NAME", "env-model")
    monkeypatch.setenv("MODEL_VERSION", "7.7.7")
    monkeypatch.setenv("PREDICTION_THRESHOLD", "2.5")
    monkeypatch.setenv("PORT", "9090")

    settings = Settings.from_env()

    assert settings.model_name == "env-model"
    assert settings.model_version == "7.7.7"
    assert settings.threshold == 2.5
    assert settings.port == 9090
