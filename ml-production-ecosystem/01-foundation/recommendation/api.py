"""Local FastAPI serving for active registered recommender model."""

from pathlib import Path
import argparse
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from .predict import recommend_top_k_from_registry
from .train import get_active_model

DEFAULT_MODEL_NAME = "movielens-popularity"
DEFAULT_REGISTRY_PATH = Path("01-foundation/registry/models.json")


class PredictionRequest(BaseModel):
    user_id: int | None = None
    top_k: int = 10


class ServingMetrics:
    def __init__(self) -> None:
        self.prediction_request_count = 0
        self.prediction_error_count = 0
        self.prediction_latency_ms_total = 0.0
        self.prediction_latency_ms_last = 0.0
        self.last_model_name: str | None = None
        self.last_model_version: str | None = None

    def record_prediction(
        self,
        latency_ms: float,
        error: bool,
        model_name: object | None = None,
        model_version: object | None = None,
    ) -> None:
        self.prediction_request_count += 1
        if error:
            self.prediction_error_count += 1
        self.prediction_latency_ms_last = round(latency_ms, 6)
        self.prediction_latency_ms_total += latency_ms
        if model_name is not None:
            self.last_model_name = str(model_name)
        if model_version is not None:
            self.last_model_version = str(model_version)

    def snapshot(self) -> dict[str, object]:
        if self.prediction_request_count == 0:
            latency_avg = 0.0
        else:
            latency_avg = round(self.prediction_latency_ms_total / self.prediction_request_count, 6)
        return {
            "prediction_request_count": self.prediction_request_count,
            "prediction_error_count": self.prediction_error_count,
            "prediction_latency_ms_avg": latency_avg,
            "prediction_latency_ms_last": self.prediction_latency_ms_last,
            "last_model_name": self.last_model_name,
            "last_model_version": self.last_model_version,
        }


def create_app(registry_path: Path = DEFAULT_REGISTRY_PATH, model_name: str = DEFAULT_MODEL_NAME) -> FastAPI:
    app = FastAPI(title="Foundation Recommender API")
    metrics = ServingMetrics()

    def active_model() -> dict[str, object]:
        model = get_active_model(registry_path, model_name)
        if model is None:
            raise HTTPException(status_code=404, detail=f"active model not found: {model_name}")
        return model

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/models/active")
    def get_active_model_endpoint() -> dict[str, object]:
        return active_model()

    @app.get("/metrics")
    def get_metrics() -> dict[str, object]:
        return metrics.snapshot()

    @app.post("/predict/v1")
    def predict(request: PredictionRequest) -> dict[str, object]:
        request_id = str(uuid4())
        started_at = perf_counter()
        model: dict[str, object] | None = None
        try:
            model = active_model()
            recommendations = recommend_top_k_from_registry(registry_path, model_name, request.top_k, request.user_id)
        except HTTPException:
            metrics.record_prediction((perf_counter() - started_at) * 1000, error=True)
            raise
        except Exception as error:
            metrics.record_prediction(
                (perf_counter() - started_at) * 1000,
                error=True,
                model_name=model.get("model_name") if model else None,
                model_version=model.get("version") if model else None,
            )
            raise HTTPException(status_code=500, detail=str(error)) from error

        metrics.record_prediction(
            (perf_counter() - started_at) * 1000,
            error=False,
            model_name=model["model_name"],
            model_version=model["version"],
        )
        return {
            "request_id": request_id,
            "model_name": model["model_name"],
            "version": model["version"],
            "recommendations": recommendations,
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve active registered foundation recommender model.")
    parser.add_argument("--registry-path", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(create_app(args.registry_path, args.model_name), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
