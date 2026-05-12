"""Predict recommendations from local popularity artifact."""

from pathlib import Path
import argparse
import json

from .artifacts import load_artifact


def recommend_top_k(artifact_path: Path, top_k: int = 10) -> list[dict[str, object]]:
    artifact = load_artifact(artifact_path)
    return artifact.model["recommendations"][:top_k]


def main() -> None:
    parser = argparse.ArgumentParser(description="Return top-k MovieLens recommendations.")
    parser.add_argument("--artifact-path", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    print(json.dumps(recommend_top_k(args.artifact_path, args.top_k), indent=2))


if __name__ == "__main__":
    main()
