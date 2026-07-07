from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from time import perf_counter

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
import uvicorn

from .predict import predict

LOGGER = logging.getLogger("{{package_name}}")


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a number") from error


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error


@dataclass(frozen=True)
class Settings:
    service_name: str = "{{project_name}}"
    model_name: str = "{{package_name}}-dummy"
    model_version: str = "0.1.0"
    threshold: float = 0.0
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            service_name=os.getenv("SERVICE_NAME", cls.service_name),
            model_name=os.getenv("MODEL_NAME", cls.model_name),
            model_version=os.getenv("MODEL_VERSION", cls.model_version),
            threshold=_float_env("PREDICTION_THRESHOLD", cls.threshold),
            host=os.getenv("HOST", cls.host),
            port=_int_env("PORT", cls.port),
            log_level=os.getenv("LOG_LEVEL", cls.log_level),
        )


class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description="Numeric feature values for the placeholder model.",
        examples=[{"age": 42.0, "balance": 120.5}],
    )


class PredictionResponse(BaseModel):
    prediction: bool
    model_name: str
    model_version: str


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    configure_logging(settings.log_level)
    app = FastAPI(title=settings.service_name, version=settings.model_version)

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = perf_counter()
        response = await call_next(request)
        duration_ms = (perf_counter() - start) * 1000
        LOGGER.info(
            "request completed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 3),
            },
        )
        return response

    @app.get("/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "model_name": settings.model_name,
            "model_version": settings.model_version,
        }

    @app.post("/predict", response_model=PredictionResponse)
    def predict_endpoint(request: PredictionRequest) -> PredictionResponse:
        score = sum(request.features.values())
        prediction = predict(request.features, threshold=settings.threshold)
        LOGGER.info(
            "prediction served",
            extra={
                "model_name": settings.model_name,
                "model_version": settings.model_version,
                "feature_count": len(request.features),
                "score": score,
            },
        )
        return PredictionResponse(
            prediction=prediction,
            model_name=settings.model_name,
            model_version=settings.model_version,
        )

    return app


app = create_app()


def main() -> None:
    settings = Settings.from_env()
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
