from fastapi import FastAPI
from pydantic import BaseModel

from .transcribe import transcribe

app = FastAPI(title="{{project_name}}")

class TranscriptionRequest(BaseModel):
    audio_uri: str
    language: str = "id"

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/predict/v1")
def predict_endpoint(request: TranscriptionRequest) -> dict[str, str]:
    return {
        "transcript": transcribe(request.audio_uri, request.language),
        "model_version": "local-dev",
    }
