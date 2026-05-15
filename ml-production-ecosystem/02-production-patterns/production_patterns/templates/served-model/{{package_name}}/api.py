from fastapi import FastAPI
from pydantic import BaseModel

from .predict import predict

app = FastAPI(title="{{project_name}}")


class PredictionRequest(BaseModel):
    features: dict[str, float]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict_endpoint(request: PredictionRequest) -> dict[str, float]:
    return {"prediction": predict(request.features)}
