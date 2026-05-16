"""Adapter from MovieLens recommender to generic prediction contracts."""

from pathlib import Path

from ml_production_ecosystem.shared.model_contracts import ModelIOContract, ModelMetadata, PredictionRequest, PredictionResponse

from .predict import recommend_top_k_from_registry
from .train import get_active_model

DEFAULT_IO_CONTRACT = ModelIOContract(
    input_schema_uri="schemas/recommendation/input.json",
    output_schema_uri="schemas/recommendation/output.json",
    task_type="ranking",
    prediction_key="recommendations",
)

class RecommenderPredictionAdapter:
    """Generic prediction port for local recommender artifacts."""

    def __init__(
        self,
        registry_path: Path,
        model_name: str = "movielens-popularity",
        io_contract: ModelIOContract = DEFAULT_IO_CONTRACT,
    ) -> None:
        self.registry_path = registry_path
        self.model_name = model_name
        self.io_contract = io_contract

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        active_model = get_active_model(self.registry_path, self.model_name)
        if active_model is None:
            raise ValueError(f"active model not found: {self.model_name}")

        predictions = []
        for record in request.records:
            top_k = int(record.get("top_k", 10) or 10)
            user_id_value = record.get("user_id")
            user_id = int(user_id_value) if user_id_value is not None else None
            predictions.append(
                recommend_top_k_from_registry(
                    registry_path=self.registry_path,
                    model_name=self.model_name,
                    top_k=top_k,
                    user_id=user_id,
                )
            )

        metadata = ModelMetadata(
            name=str(active_model["model_name"]),
            version=str(active_model["version"]),
            framework="local-json",
            io_contract=self.io_contract,
            artifact_uri=str(active_model["artifact_uri"]),
        )
        return PredictionResponse(
            predictions=predictions,
            model=metadata,
            request_id=request.request_id,
            metadata={"adapter": "ml_production_ecosystem.recommendation.prediction_adapter"},
        )
