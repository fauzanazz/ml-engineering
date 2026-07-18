from {{package_name}}.api import PredictionRequest, health, predict_endpoint


def test_api_health_and_prediction_contract() -> None:
    assert health() == {"status": "ok"}
    response = predict_endpoint(PredictionRequest(features={"x": 1.0}))
    assert "prediction" in response
