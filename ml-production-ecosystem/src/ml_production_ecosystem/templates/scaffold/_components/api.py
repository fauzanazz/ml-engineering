from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

try:
    from .predict import predict
except ImportError:
    def predict(features: dict[str, float]) -> object:
        return sum(features.values()) if features else 0.0

app = FastAPI(title="{{project_name}}")


class PredictionRequest(BaseModel):
    features: dict[str, float]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict_endpoint(request: PredictionRequest) -> dict[str, object]:
    return {"prediction": predict(request.features)}
