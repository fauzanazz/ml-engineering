"""Local artifact helpers for foundation recommender."""

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any


@dataclass(frozen=True)
class RecommendationArtifact:
    """Loaded recommendation artifact."""

    model: dict[str, Any]
    metadata: dict[str, Any]
    metrics: dict[str, Any]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_artifact(artifact_path: Path) -> RecommendationArtifact:
    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact not found: {artifact_path}. Run training first.")
    return RecommendationArtifact(
        model=load_json(artifact_path / "model.json"),
        metadata=load_json(artifact_path / "metadata.json"),
        metrics=load_json(artifact_path / "metrics.json"),
    )
