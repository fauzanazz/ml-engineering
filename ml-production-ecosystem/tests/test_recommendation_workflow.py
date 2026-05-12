from pathlib import Path

import pytest

from recommendation.artifacts import load_artifact
from recommendation.data import DatasetValidationError, validate_movielens_files
from recommendation.predict import recommend_top_k
from recommendation.train import train_popularity_recommender

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "recommendation"


def test_validate_movielens_files_rejects_missing_columns(tmp_path: Path) -> None:
    ratings = tmp_path / "ratings.csv"
    movies = tmp_path / "movies.csv"
    ratings.write_text("userId,movieId\n1,1\n")
    movies.write_text("movieId,title\n1,Toy Story\n")

    with pytest.raises(DatasetValidationError):
        validate_movielens_files(ratings, movies)


def test_train_popularity_recommender_writes_artifact(tmp_path: Path) -> None:
    result = train_popularity_recommender(
        ratings_path=FIXTURE_DIR / "ratings.csv",
        movies_path=FIXTURE_DIR / "movies.csv",
        artifact_dir=tmp_path,
        version="test-v1",
        min_rating=4.0,
    )

    assert result.model_name == "movielens-popularity"
    assert result.version == "test-v1"
    assert (Path(result.uri) / "model.json").exists()
    assert (Path(result.uri) / "metadata.json").exists()
    assert (Path(result.uri) / "metrics.json").exists()

    artifact = load_artifact(Path(result.uri))
    assert artifact.metadata["dataset"] == "MovieLens 25M"
    assert artifact.metrics["ratings_rows"] == 8
    assert artifact.model["recommendations"][0]["movieId"] == 1


def test_recommend_top_k_returns_sorted_recommendations(tmp_path: Path) -> None:
    result = train_popularity_recommender(
        ratings_path=FIXTURE_DIR / "ratings.csv",
        movies_path=FIXTURE_DIR / "movies.csv",
        artifact_dir=tmp_path,
        version="test-v1",
        min_rating=4.0,
    )

    recommendations = recommend_top_k(Path(result.uri), top_k=2)

    assert [item["movieId"] for item in recommendations] == [1, 3]
    assert recommendations[0]["title"] == "Toy Story (1995)"
    assert recommendations[0]["score"] >= recommendations[1]["score"]


def test_gitignore_excludes_local_data_and_artifacts() -> None:
    gitignore = Path(__file__).resolve().parents[1] / ".gitignore"
    text = gitignore.read_text()

    assert "data/" in text
    assert "artifacts/" in text
