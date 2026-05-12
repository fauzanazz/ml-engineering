from shared.deployment import DeploymentResult
from shared.model_storage import ModelArtifact
from shared.monitoring import MonitoringResult
from shared.observability import MetricPoint


def test_shared_contracts_are_importable() -> None:
    artifact = ModelArtifact(name="demo", version="v1", uri="models/demo/v1")
    deployment = DeploymentResult(model_name="demo", model_version="v1", endpoint="local://demo")
    metric = MetricPoint(name="prediction_latency_ms", value=12.5, tags={"model": "demo"})
    result = MonitoringResult(check_name="data_schema", passed=True, message="ok")

    assert artifact.name == "demo"
    assert deployment.endpoint == "local://demo"
    assert metric.value == 12.5
    assert result.passed is True
