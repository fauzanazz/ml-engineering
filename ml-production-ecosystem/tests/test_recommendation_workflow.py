from pathlib import Path

import pytest

from recommendation.artifacts import load_artifact, make_run_dir
from recommendation.data import DatasetValidationError, load_three_way_ratings_split, validate_movielens_files
from recommendation.evaluate import evaluate_popularity_recommender
from recommendation.predict import recommend_top_k
from recommendation.train import (
    train_collaborative_filtering_recommender,
    train_matrix_factorization_recommender,
    train_popularity_recommender,
)

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


def test_load_three_way_ratings_split_orders_by_timestamp_without_overlap() -> None:
    train, val, test = load_three_way_ratings_split(
        FIXTURE_DIR / "ratings.csv",
        val_size=0.25,
        test_size=0.25,
    )

    assert [row["timestamp"] for row in train] == [100, 101, 102, 103]
    assert [row["timestamp"] for row in val] == [104, 105]
    assert [row["timestamp"] for row in test] == [106, 107]
    assert {id(row) for row in train}.isdisjoint({id(row) for row in val})
    assert {id(row) for row in train}.isdisjoint({id(row) for row in test})
    assert {id(row) for row in val}.isdisjoint({id(row) for row in test})


def test_evaluate_popularity_recommender_reports_ranking_metrics(tmp_path: Path) -> None:
    result = train_popularity_recommender(
        ratings_path=FIXTURE_DIR / "ratings.csv",
        movies_path=FIXTURE_DIR / "movies.csv",
        artifact_dir=tmp_path,
        version="test-v1",
        min_rating=4.0,
    )

    metrics = evaluate_popularity_recommender(
        artifact_path=Path(result.uri),
        ratings_path=FIXTURE_DIR / "ratings.csv",
        top_k=2,
        min_relevant_rating=4.0,
    )

    assert metrics.precision_at_k == 0.625
    assert metrics.recall_at_k == 1.0
    assert metrics.hit_rate_at_k == 1.0
    assert metrics.coverage == 0.5


def test_make_run_dir_adds_timestamp_suffix_when_run_id_absent(tmp_path: Path) -> None:
    run_dir = make_run_dir(tmp_path)

    assert run_dir.parent == tmp_path
    assert len(run_dir.name.split("-")) == 2
    assert not run_dir.exists()


def test_collaborative_filtering_returns_user_specific_unseen_movies(tmp_path: Path) -> None:
    result = train_collaborative_filtering_recommender(
        ratings_path=FIXTURE_DIR / "ratings.csv",
        movies_path=FIXTURE_DIR / "movies.csv",
        artifact_dir=tmp_path,
        version="cf-v1",
        min_rating=4.0,
    )

    recommendations = recommend_top_k(Path(result.uri), top_k=2, user_id=10)

    assert result.model_name == "movielens-item-collaborative-filtering"
    assert [item["movieId"] for item in recommendations] == [3]
    assert recommendations[0]["title"] == "Heat (1995)"
    assert recommendations[0]["reason"] == "similar_to_user_history"


def test_matrix_factorization_returns_user_specific_unseen_movies(tmp_path: Path) -> None:
    result = train_matrix_factorization_recommender(
        ratings_path=FIXTURE_DIR / "ratings.csv",
        movies_path=FIXTURE_DIR / "movies.csv",
        artifact_dir=tmp_path,
        version="mf-v1",
        factors=2,
        epochs=80,
        learning_rate=0.01,
        regularization=0.01,
    )

    recommendations = recommend_top_k(Path(result.uri), top_k=2, user_id=10)

    assert result.model_name == "movielens-matrix-factorization"
    assert recommendations
    assert all(item["movieId"] not in {1, 2} for item in recommendations)
    assert all("predicted_rating" in item for item in recommendations)
