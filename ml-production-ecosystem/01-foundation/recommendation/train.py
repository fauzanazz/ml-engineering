"""Train baseline MovieLens popularity recommender."""

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import argparse
import csv
from datetime import UTC, datetime
import random

from .artifacts import write_json
from .data import validate_movielens_files

MODEL_NAME = "movielens-popularity"
COLLABORATIVE_FILTERING_MODEL_NAME = "movielens-item-collaborative-filtering"
MATRIX_FACTORIZATION_MODEL_NAME = "movielens-matrix-factorization"


@dataclass(frozen=True)
class TrainingResult:
    model_name: str
    version: str
    uri: str
    metrics_uri: str


def _load_movies(movies_path: Path) -> dict[int, str]:
    movies: dict[int, str] = {}
    with movies_path.open(newline="") as file:
        for row in csv.DictReader(file):
            movies[int(row["movieId"])] = row["title"]
    return movies


def _load_ratings(ratings_path: Path) -> list[dict[str, int | float]]:
    with ratings_path.open(newline="") as file:
        return [
            {
                "userId": int(row["userId"]),
                "movieId": int(row["movieId"]),
                "rating": float(row["rating"]),
                "timestamp": int(row["timestamp"]),
            }
            for row in csv.DictReader(file)
        ]


def _write_recommender_artifact(
    output_dir: Path,
    model: dict[str, object],
    metadata: dict[str, object],
    metrics: dict[str, object],
) -> None:
    write_json(output_dir / "model.json", model)
    write_json(output_dir / "metadata.json", metadata)
    write_json(output_dir / "metrics.json", metrics)


def train_popularity_recommender(
    ratings_path: Path,
    movies_path: Path,
    artifact_dir: Path,
    version: str | None = None,
    min_rating: float = 4.0,
) -> TrainingResult:
    validate_movielens_files(ratings_path, movies_path)
    version = version or datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    output_dir = artifact_dir / "recommendation" / version
    movies = _load_movies(movies_path)

    stats: dict[int, dict[str, float]] = {}
    ratings_rows = 0
    users: set[str] = set()
    with ratings_path.open(newline="") as file:
        for row in csv.DictReader(file):
            ratings_rows += 1
            users.add(row["userId"])
            movie_id = int(row["movieId"])
            rating = float(row["rating"])
            item = stats.setdefault(movie_id, {"rating_count": 0.0, "rating_sum": 0.0, "positive_count": 0.0})
            item["rating_count"] += 1
            item["rating_sum"] += rating
            if rating >= min_rating:
                item["positive_count"] += 1

    recommendations = []
    for movie_id, item in stats.items():
        count = item["rating_count"]
        avg = item["rating_sum"] / count
        score = item["positive_count"] * avg
        recommendations.append(
            {
                "movieId": movie_id,
                "title": movies.get(movie_id, f"movie:{movie_id}"),
                "score": round(score, 6),
                "rating_count": int(count),
                "average_rating": round(avg, 6),
                "positive_count": int(item["positive_count"]),
            }
        )
    recommendations.sort(key=lambda item: (item["score"], item["rating_count"], item["average_rating"]), reverse=True)

    model = {"model_name": MODEL_NAME, "version": version, "recommendations": recommendations}
    metadata = {
        "model_name": MODEL_NAME,
        "version": version,
        "dataset": "MovieLens 25M",
        "ratings_path": str(ratings_path),
        "movies_path": str(movies_path),
        "trained_at": datetime.now(UTC).isoformat(),
        "min_rating": min_rating,
    }
    metrics = {
        "ratings_rows": ratings_rows,
        "unique_users": len(users),
        "unique_movies": len(stats),
        "candidate_count": len(recommendations),
    }

    write_json(output_dir / "model.json", model)
    write_json(output_dir / "metadata.json", metadata)
    write_json(output_dir / "metrics.json", metrics)
    return TrainingResult(MODEL_NAME, version, str(output_dir), str(output_dir / "metrics.json"))


def train_collaborative_filtering_recommender(
    ratings_path: Path,
    movies_path: Path,
    artifact_dir: Path,
    version: str | None = None,
    min_rating: float = 4.0,
) -> TrainingResult:
    validate_movielens_files(ratings_path, movies_path)
    version = version or datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    output_dir = artifact_dir / "recommendation" / version
    movies = _load_movies(movies_path)
    ratings = _load_ratings(ratings_path)

    user_history: dict[int, dict[int, float]] = {}
    liked_by_movie: dict[int, set[int]] = {}
    for row in ratings:
        user_id = int(row["userId"])
        movie_id = int(row["movieId"])
        rating = float(row["rating"])
        user_history.setdefault(user_id, {})[movie_id] = rating
        if rating >= min_rating:
            liked_by_movie.setdefault(movie_id, set()).add(user_id)

    similarities: dict[str, float] = {}
    for left_movie_id, right_movie_id in combinations(sorted(liked_by_movie), 2):
        left_users = liked_by_movie[left_movie_id]
        right_users = liked_by_movie[right_movie_id]
        overlap = len(left_users & right_users)
        if overlap == 0:
            continue
        union = len(left_users | right_users)
        similarities[f"{left_movie_id}:{right_movie_id}"] = round(overlap / union, 6)

    model = {
        "model_name": COLLABORATIVE_FILTERING_MODEL_NAME,
        "version": version,
        "movies": {str(movie_id): title for movie_id, title in movies.items()},
        "user_history": {str(user_id): {str(movie_id): rating for movie_id, rating in history.items()} for user_id, history in user_history.items()},
        "item_similarities": similarities,
        "min_rating": min_rating,
    }
    metadata = {
        "model_name": COLLABORATIVE_FILTERING_MODEL_NAME,
        "version": version,
        "dataset": "MovieLens 25M",
        "ratings_path": str(ratings_path),
        "movies_path": str(movies_path),
        "trained_at": datetime.now(UTC).isoformat(),
        "min_rating": min_rating,
    }
    metrics = {
        "ratings_rows": len(ratings),
        "unique_users": len(user_history),
        "unique_movies": len(movies),
        "similarity_count": len(similarities),
    }
    _write_recommender_artifact(output_dir, model, metadata, metrics)
    return TrainingResult(COLLABORATIVE_FILTERING_MODEL_NAME, version, str(output_dir), str(output_dir / "metrics.json"))


def train_matrix_factorization_recommender(
    ratings_path: Path,
    movies_path: Path,
    artifact_dir: Path,
    version: str | None = None,
    factors: int = 8,
    epochs: int = 20,
    learning_rate: float = 0.01,
    regularization: float = 0.02,
) -> TrainingResult:
    validate_movielens_files(ratings_path, movies_path)
    if factors < 1:
        raise ValueError(f"factors must be positive, got {factors}")
    if epochs < 1:
        raise ValueError(f"epochs must be positive, got {epochs}")

    version = version or datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    output_dir = artifact_dir / "recommendation" / version
    movies = _load_movies(movies_path)
    ratings = _load_ratings(ratings_path)
    global_mean = sum(float(row["rating"]) for row in ratings) / len(ratings)

    rng = random.Random(42)
    user_factors: dict[int, list[float]] = {}
    movie_factors: dict[int, list[float]] = {}
    user_history: dict[int, list[int]] = {}
    for row in ratings:
        user_id = int(row["userId"])
        movie_id = int(row["movieId"])
        user_factors.setdefault(user_id, [rng.uniform(-0.05, 0.05) for _ in range(factors)])
        movie_factors.setdefault(movie_id, [rng.uniform(-0.05, 0.05) for _ in range(factors)])
        user_history.setdefault(user_id, []).append(movie_id)

    for _ in range(epochs):
        for row in ratings:
            user_id = int(row["userId"])
            movie_id = int(row["movieId"])
            rating = float(row["rating"])
            user_vector = user_factors[user_id]
            movie_vector = movie_factors[movie_id]
            predicted = global_mean + sum(user_vector[index] * movie_vector[index] for index in range(factors))
            error = rating - predicted
            for index in range(factors):
                user_value = user_vector[index]
                movie_value = movie_vector[index]
                user_vector[index] += learning_rate * (error * movie_value - regularization * user_value)
                movie_vector[index] += learning_rate * (error * user_value - regularization * movie_value)

    model = {
        "model_name": MATRIX_FACTORIZATION_MODEL_NAME,
        "version": version,
        "movies": {str(movie_id): title for movie_id, title in movies.items()},
        "global_mean": round(global_mean, 6),
        "user_factors": {str(user_id): vector for user_id, vector in user_factors.items()},
        "movie_factors": {str(movie_id): vector for movie_id, vector in movie_factors.items()},
        "user_history": {str(user_id): sorted(set(movie_ids)) for user_id, movie_ids in user_history.items()},
    }
    metadata = {
        "model_name": MATRIX_FACTORIZATION_MODEL_NAME,
        "version": version,
        "dataset": "MovieLens 25M",
        "ratings_path": str(ratings_path),
        "movies_path": str(movies_path),
        "trained_at": datetime.now(UTC).isoformat(),
        "factors": factors,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "regularization": regularization,
    }
    metrics = {
        "ratings_rows": len(ratings),
        "unique_users": len(user_factors),
        "unique_movies": len(movie_factors),
        "factor_count": factors,
    }
    _write_recommender_artifact(output_dir, model, metadata, metrics)
    return TrainingResult(MATRIX_FACTORIZATION_MODEL_NAME, version, str(output_dir), str(output_dir / "metrics.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MovieLens popularity recommender.")
    parser.add_argument("--ratings-path", type=Path, required=True)
    parser.add_argument("--movies-path", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--version")
    parser.add_argument("--min-rating", type=float, default=4.0)
    parser.add_argument("--model", choices=["popularity", "collaborative-filtering", "matrix-factorization"], default="popularity")
    parser.add_argument("--factors", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--regularization", type=float, default=0.02)
    args = parser.parse_args()
    if args.model == "collaborative-filtering":
        result = train_collaborative_filtering_recommender(
            args.ratings_path, args.movies_path, args.artifact_dir, args.version, args.min_rating
        )
    elif args.model == "matrix-factorization":
        result = train_matrix_factorization_recommender(
            args.ratings_path,
            args.movies_path,
            args.artifact_dir,
            args.version,
            args.factors,
            args.epochs,
            args.learning_rate,
            args.regularization,
        )
    else:
        result = train_popularity_recommender(args.ratings_path, args.movies_path, args.artifact_dir, args.version, args.min_rating)
    print(result)


if __name__ == "__main__":
    main()
