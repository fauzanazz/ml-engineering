from shared.deployment import DeploymentResult
from shared.lifecycle import LifecycleRun, ReleaseDecision, RollbackPlan
from shared.model_contracts import (
    ModelIOContract,
    ModelMetadata,
    PredictionRequest,
    PredictionResponse,
)
from shared.model_storage import ModelArtifact
from shared.monitoring import MonitoringResult
from shared.observability import MetricPoint
from shared.platform import CloudResourceRef, InfrastructurePlan, SecretRef


def test_shared_contracts_are_importable() -> None:
    artifact = ModelArtifact(name="demo", version="v1", uri="models/demo/v1")
    deployment = DeploymentResult(model_name="demo", model_version="v1", endpoint="local://demo")
    metric = MetricPoint(name="prediction_latency_ms", value=12.5, tags={"model": "demo"})
    result = MonitoringResult(check_name="data_schema", passed=True, message="ok")

    assert artifact.name == "demo"
    assert deployment.endpoint == "local://demo"
    assert metric.value == 12.5
    assert result.passed is True


def test_model_contracts_are_model_agnostic() -> None:
    io_contract = ModelIOContract(
        input_schema_uri="schemas/recommender/input.json",
        output_schema_uri="schemas/recommender/output.json",
        task_type="ranking",
        prediction_key="recommendations",
    )
    metadata = ModelMetadata(
        name="demo-ranker",
        version="v1",
        framework="sklearn",
        io_contract=io_contract,
        artifact_uri="models/demo-ranker/v1",
    )
    request = PredictionRequest(records=[{"user_id": 42, "limit": 5}], request_id="req-1")
    response = PredictionResponse(
        predictions=[[{"item_id": 7, "score": 0.9}]],
        model=metadata,
        request_id=request.request_id,
    )

    assert request.records[0]["user_id"] == 42
    assert response.model.io_contract.task_type == "ranking"
    assert response.predictions[0][0]["score"] == 0.9


def test_platform_contracts_store_secret_references_only() -> None:
    secret = SecretRef(
        provider="aws",
        name="prod/ml/model-registry-token",
        injection_target="MODEL_REGISTRY_TOKEN",
        policy_ref="policy/ml-serving-read-secret",
    )
    resource = CloudResourceRef(
        provider="aws",
        kind="object-storage",
        name="model-artifacts",
        uri="s3://example-model-artifacts",
    )
    plan = InfrastructurePlan(provider="aws", resources=(resource,), secrets=(secret,))

    assert plan.secrets[0].name == "prod/ml/model-registry-token"
    assert plan.secrets[0].injection_target == "MODEL_REGISTRY_TOKEN"
    assert "token-value" not in repr(plan)


def test_lifecycle_contracts_are_provider_neutral() -> None:
    io_contract = ModelIOContract(
        input_schema_uri="schemas/generic/input.json",
        output_schema_uri="schemas/generic/output.json",
        task_type="classification",
        prediction_key="label",
    )
    candidate = ModelMetadata(name="classifier", version="v2", framework="xgboost", io_contract=io_contract)
    previous = ModelMetadata(name="classifier", version="v1", framework="xgboost", io_contract=io_contract)
    run = LifecycleRun(
        run_id="run-123",
        workflow="retraining",
        status="passed",
        model=candidate,
        metrics={"accuracy": 0.91},
        report_uri="reports/run-123.json",
    )
    release = ReleaseDecision(approved=True, reason="quality gate passed", model=candidate)
    rollback = RollbackPlan(current_model=candidate, target_model=previous, reason="latency regression")

    assert run.workflow == "retraining"
    assert run.model == candidate
    assert release.approved is True
    assert rollback.target_model.version == "v1"
