"""Predict recommendations from local popularity artifact."""

from pathlib import Path
import argparse
import json

from .artifacts import load_artifact
from .train import get_active_model


def _similarity_for_pair(similarities: dict[str, float], left_movie_id: int, right_movie_id: int) -> float:
    low, high = sorted((left_movie_id, right_movie_id))
    return float(similarities.get(f"{low}:{high}", 0.0))


def _recommend_collaborative_filtering(model: dict[str, object], user_id: int, top_k: int) -> list[dict[str, object]]:
    user_history = model["user_history"]
    movies = model["movies"]
    similarities = model["item_similarities"]
    history = {int(movie_id): float(rating) for movie_id, rating in user_history.get(str(user_id), {}).items()}
    seen = set(history)
    candidates = []
    for movie_id_text, title in movies.items():
        movie_id = int(movie_id_text)
        if movie_id in seen:
            continue
        score = sum(_similarity_for_pair(similarities, seen_movie_id, movie_id) * rating for seen_movie_id, rating in history.items())
        if score <= 0:
            continue
        candidates.append({"movieId": movie_id, "title": title, "score": round(score, 6), "reason": "similar_to_user_history"})
    candidates.sort(key=lambda item: (item["score"], item["movieId"]), reverse=True)
    return candidates[:top_k]


def _recommend_matrix_factorization(model: dict[str, object], user_id: int, top_k: int) -> list[dict[str, object]]:
    user_vector = model["user_factors"].get(str(user_id))
    if user_vector is None:
        return []
    seen = {int(movie_id) for movie_id in model["user_history"].get(str(user_id), [])}
    global_mean = float(model["global_mean"])
    movies = model["movies"]
    candidates = []
    for movie_id_text, movie_vector in model["movie_factors"].items():
        movie_id = int(movie_id_text)
        if movie_id in seen:
            continue
        predicted_rating = global_mean + sum(float(user_vector[index]) * float(movie_vector[index]) for index in range(len(user_vector)))
        candidates.append(
            {
                "movieId": movie_id,
                "title": movies.get(movie_id_text, f"movie:{movie_id}"),
                "predicted_rating": round(predicted_rating, 6),
                "score": round(predicted_rating, 6),
                "reason": "matrix_factorization_score",
            }
        )
    candidates.sort(key=lambda item: (item["score"], item["movieId"]), reverse=True)
    return candidates[:top_k]


def recommend_top_k(artifact_path: Path, top_k: int = 10, user_id: int | None = None) -> list[dict[str, object]]:
    artifact = load_artifact(artifact_path)
    model_name = artifact.model.get("model_name")
    if model_name == "movielens-item-collaborative-filtering":
        if user_id is None:
            raise ValueError("user_id is required for collaborative filtering recommendations")
        return _recommend_collaborative_filtering(artifact.model, user_id, top_k)
    if model_name == "movielens-matrix-factorization":
        if user_id is None:
            raise ValueError("user_id is required for matrix factorization recommendations")
        return _recommend_matrix_factorization(artifact.model, user_id, top_k)
    return artifact.model["recommendations"][:top_k]


def recommend_top_k_from_registry(
    registry_path: Path,
    model_name: str = "movielens-popularity",
    top_k: int = 10,
    user_id: int | None = None,
) -> list[dict[str, object]]:
    active_model = get_active_model(registry_path, model_name)
    if active_model is None:
        raise ValueError(f"active model not found: {model_name}")
    return recommend_top_k(Path(str(active_model["artifact_uri"])), top_k, user_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Return top-k MovieLens recommendations.")
    parser.add_argument("--artifact-path", type=Path)
    parser.add_argument("--registry-path", type=Path)
    parser.add_argument("--model-name", default="movielens-popularity")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--user-id", type=int)
    args = parser.parse_args()
    if args.registry_path:
        recommendations = recommend_top_k_from_registry(args.registry_path, args.model_name, args.top_k, args.user_id)
    elif args.artifact_path:
        recommendations = recommend_top_k(args.artifact_path, args.top_k, args.user_id)
    else:
        parser.error("--artifact-path or --registry-path is required")
    print(json.dumps(recommendations, indent=2))


if __name__ == "__main__":
    main()
