"""Ranking evaluation helpers for baseline recommenders."""

from dataclasses import dataclass
from pathlib import Path
import csv

from .artifacts import load_artifact


@dataclass(frozen=True)
class RankingMetrics:
    precision_at_k: float
    recall_at_k: float
    hit_rate_at_k: float
    coverage: float


def _load_relevant_by_user(ratings_path: Path, min_relevant_rating: float) -> tuple[dict[int, set[int]], set[int]]:
    relevant_by_user: dict[int, set[int]] = {}
    all_movies: set[int] = set()
    with ratings_path.open(newline="") as file:
        for row in csv.DictReader(file):
            user_id = int(row["userId"])
            movie_id = int(row["movieId"])
            rating = float(row["rating"])
            all_movies.add(movie_id)
            if rating >= min_relevant_rating:
                relevant_by_user.setdefault(user_id, set()).add(movie_id)
    return relevant_by_user, all_movies


def evaluate_popularity_recommender(
    artifact_path: Path,
    ratings_path: Path,
    top_k: int = 10,
    min_relevant_rating: float = 4.0,
) -> RankingMetrics:
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")

    artifact = load_artifact(artifact_path)
    recommended_ids = [int(item["movieId"]) for item in artifact.model["recommendations"][:top_k]]
    recommended_set = set(recommended_ids)
    relevant_by_user, all_movies = _load_relevant_by_user(ratings_path, min_relevant_rating)

    if not relevant_by_user:
        return RankingMetrics(precision_at_k=0.0, recall_at_k=0.0, hit_rate_at_k=0.0, coverage=0.0)

    precision_sum = 0.0
    recall_sum = 0.0
    hit_count = 0
    for relevant in relevant_by_user.values():
        hits = len(recommended_set & relevant)
        precision_sum += hits / top_k
        recall_sum += hits / len(relevant)
        if hits > 0:
            hit_count += 1

    user_count = len(relevant_by_user)
    coverage = len(recommended_set) / len(all_movies) if all_movies else 0.0
    return RankingMetrics(
        precision_at_k=precision_sum / user_count,
        recall_at_k=recall_sum / user_count,
        hit_rate_at_k=hit_count / user_count,
        coverage=coverage,
    )
