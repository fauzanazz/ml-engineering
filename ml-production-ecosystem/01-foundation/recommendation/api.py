"""Local FastAPI serving for active registered recommender model."""

from pathlib import Path
import argparse

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


def create_app(registry_path: Path = DEFAULT_REGISTRY_PATH, model_name: str = DEFAULT_MODEL_NAME) -> FastAPI:
    app = FastAPI(title="Foundation Recommender API")

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

    @app.post("/predict/v1")
    def predict(request: PredictionRequest) -> dict[str, object]:
        model = active_model()
        recommendations = recommend_top_k_from_registry(registry_path, model_name, request.top_k, request.user_id)
        return {
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
