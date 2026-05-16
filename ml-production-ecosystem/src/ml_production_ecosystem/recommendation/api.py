"""Local FastAPI serving for active registered recommender model."""

from pathlib import Path
import argparse
from datetime import UTC, datetime
import json
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
import uvicorn

from .artifacts import load_artifact
from .predict import recommend_top_k_from_registry
from .train import get_active_model

DEFAULT_MODEL_NAME = "movielens-popularity"
DEFAULT_REGISTRY_PATH = Path("01-foundation/registry/models.json")
DEFAULT_PREDICTION_LOG_PATH = Path("01-foundation/logs/predictions.jsonl")


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

    def prometheus_text(self) -> str:
        model_name = self.last_model_name or "unknown"
        model_version = self.last_model_version or "unknown"
        labels = f'{{model_name="{model_name}",model_version="{model_version}"}}'
        return "\n".join(
            [
                f"foundation_prediction_requests_total{labels} {self.prediction_request_count}",
                f"foundation_prediction_errors_total{labels} {self.prediction_error_count}",
                f"foundation_prediction_latency_ms_sum{labels} {round(self.prediction_latency_ms_total, 6)}",
                f"foundation_prediction_latency_ms_count{labels} {self.prediction_request_count}",
                f"foundation_prediction_latency_ms_last{labels} {self.prediction_latency_ms_last}",
                "",
            ]
        )


def _append_prediction_log(prediction_log_path: Path, row: dict[str, object]) -> None:
    prediction_log_path.parent.mkdir(parents=True, exist_ok=True)
    with prediction_log_path.open("a") as file:
        file.write(json.dumps(row, sort_keys=True) + "\n")


def _read_prediction_logs(prediction_log_path: Path) -> list[dict[str, object]]:
    if not prediction_log_path.exists():
        return []
    return [json.loads(line) for line in prediction_log_path.read_text().splitlines() if line.strip()]


def _movie_ids(recommendations: list[dict[str, object]]) -> list[int]:
    return [int(item["movieId"]) for item in recommendations]


def create_app(
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    model_name: str = DEFAULT_MODEL_NAME,
    prediction_log_path: Path = DEFAULT_PREDICTION_LOG_PATH,
) -> FastAPI:
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

    @app.get("/metrics.json")
    def get_metrics_json() -> dict[str, object]:
        return metrics.snapshot()

    @app.get("/metrics")
    def get_metrics() -> Response:
        return Response(content=metrics.prometheus_text(), media_type="text/plain")

    @app.get("/drift")
    def get_drift(sample_limit: int = 10) -> dict[str, object]:
        model = active_model()
        artifact = load_artifact(Path(str(model["artifact_uri"])))
        baseline_ids = _movie_ids(artifact.model["recommendations"][:sample_limit])
        rows = [row for row in _read_prediction_logs(prediction_log_path) if row.get("status") == "success"]
        latest_rows = rows[-sample_limit:]
        sample_ids = [
            int(movie_id)
            for row in latest_rows
            for movie_id in row.get("recommendation_movie_ids", [])
        ][:sample_limit]
        if not sample_ids:
            drift_score = 0.0
        else:
            baseline_set = set(baseline_ids)
            drift_score = 1.0 - (sum(1 for movie_id in sample_ids if movie_id in baseline_set) / len(sample_ids))
        return {
            "model_name": model["model_name"],
            "version": model["version"],
            "sample_size": len(sample_ids),
            "baseline_size": sample_limit,
            "drift_score": round(drift_score, 6),
        }

    @app.post("/predict/v1")
    def predict(request: PredictionRequest) -> dict[str, object]:
        request_id = str(uuid4())
        requested_at = datetime.now(UTC).isoformat()
        started_at = perf_counter()
        model: dict[str, object] | None = None
        try:
            model = active_model()
            recommendations = recommend_top_k_from_registry(registry_path, model_name, request.top_k, request.user_id)
        except HTTPException as error:
            latency_ms = (perf_counter() - started_at) * 1000
            metrics.record_prediction(latency_ms, error=True)
            _append_prediction_log(
                prediction_log_path,
                {
                    "request_id": request_id,
                    "requested_at": requested_at,
                    "user_id": request.user_id,
                    "top_k": request.top_k,
                    "model_name": None,
                    "model_version": None,
                    "recommendation_movie_ids": [],
                    "status": "error",
                    "latency_ms": round(latency_ms, 6),
                    "error": str(error.detail),
                },
            )
            raise
        except Exception as error:
            latency_ms = (perf_counter() - started_at) * 1000
            metrics.record_prediction(
                latency_ms,
                error=True,
                model_name=model.get("model_name") if model else None,
                model_version=model.get("version") if model else None,
            )
            _append_prediction_log(
                prediction_log_path,
                {
                    "request_id": request_id,
                    "requested_at": requested_at,
                    "user_id": request.user_id,
                    "top_k": request.top_k,
                    "model_name": model.get("model_name") if model else None,
                    "model_version": model.get("version") if model else None,
                    "recommendation_movie_ids": [],
                    "status": "error",
                    "latency_ms": round(latency_ms, 6),
                    "error": str(error),
                },
            )
            raise HTTPException(status_code=500, detail=str(error)) from error

        latency_ms = (perf_counter() - started_at) * 1000
        metrics.record_prediction(
            latency_ms,
            error=False,
            model_name=model["model_name"],
            model_version=model["version"],
        )
        _append_prediction_log(
            prediction_log_path,
            {
                "request_id": request_id,
                "requested_at": requested_at,
                "user_id": request.user_id,
                "top_k": request.top_k,
                "model_name": model["model_name"],
                "model_version": model["version"],
                "recommendation_movie_ids": _movie_ids(recommendations),
                "status": "success",
                "latency_ms": round(latency_ms, 6),
                "error": None,
            },
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
    parser.add_argument("--prediction-log-path", type=Path, default=DEFAULT_PREDICTION_LOG_PATH)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(
        create_app(args.registry_path, args.model_name, args.prediction_log_path),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
