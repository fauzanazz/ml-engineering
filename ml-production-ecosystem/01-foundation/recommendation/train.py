"""Train baseline MovieLens popularity recommender."""

from dataclasses import dataclass
from pathlib import Path
import argparse
import csv
from datetime import UTC, datetime

from .artifacts import write_json
from .data import validate_movielens_files

MODEL_NAME = "movielens-popularity"


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MovieLens popularity recommender.")
    parser.add_argument("--ratings-path", type=Path, required=True)
    parser.add_argument("--movies-path", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--version")
    parser.add_argument("--min-rating", type=float, default=4.0)
    args = parser.parse_args()
    result = train_popularity_recommender(args.ratings_path, args.movies_path, args.artifact_dir, args.version, args.min_rating)
    print(result)


if __name__ == "__main__":
    main()
