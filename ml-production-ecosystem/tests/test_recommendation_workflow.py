from pathlib import Path
import json

import pytest

from recommendation.artifacts import load_artifact, make_run_dir
from recommendation.data import DatasetValidationError, load_three_way_ratings_split, validate_movielens_files
from recommendation.evaluate import evaluate_popularity_recommender
from recommendation.predict import recommend_top_k, recommend_top_k_from_registry
from recommendation.train import (
    get_active_model,
    get_model_version,
    list_experiment_runs,
    list_model_versions,
    register_model_version,
    set_active_model,
    train_collaborative_filtering_recommender,
    train_matrix_factorization_recommender,
    train_popularity_recommender,
    train_recommender_from_config,
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


def test_train_recommender_from_config_writes_versioned_artifact(tmp_path: Path) -> None:
    config_path = tmp_path / "foundation-recommender.yaml"
    config_path.write_text(
        f"""
pipeline:
  name: foundation-recommender
  version: config-test-v1

dataset:
  ratings_path: {FIXTURE_DIR / "ratings.csv"}
  movies_path: {FIXTURE_DIR / "movies.csv"}

model:
  type: popularity
  hyperparams:
    min_rating: 4.5

artifacts:
  artifact_dir: {tmp_path / "artifacts"}

experiments:
  tracking_dir: {tmp_path / "experiments" / "runs"}
  run_id: config-test-run
""".strip()
    )

    result = train_recommender_from_config(config_path)

    assert result.model_name == "movielens-popularity"
    assert result.version == "config-test-v1"
    assert (Path(result.uri) / "model.json").exists()
    assert (Path(result.uri) / "metadata.json").exists()
    assert (Path(result.uri) / "metrics.json").exists()

    artifact = load_artifact(Path(result.uri))
    assert artifact.metadata["min_rating"] == 4.5
    assert artifact.model["version"] == "config-test-v1"

    run_dir = tmp_path / "experiments" / "runs" / "config-test-run"
    assert (run_dir / "config.yaml").read_text() == config_path.read_text()
    assert json.loads((run_dir / "params.json").read_text()) == {"model_type": "popularity", "min_rating": 4.5}
    assert json.loads((run_dir / "metrics.json").read_text()) == artifact.metrics
    assert json.loads((run_dir / "artifact.json").read_text()) == {
        "artifact_uri": result.uri,
        "metrics_uri": result.metrics_uri,
    }
    run = json.loads((run_dir / "run.json").read_text())
    assert run["run_id"] == "config-test-run"
    assert run["model_name"] == "movielens-popularity"
    assert run["version"] == "config-test-v1"
    assert run["artifact_uri"] == result.uri
    assert run["metrics_uri"] == result.metrics_uri
    assert run["status"] == "completed"
    assert "created_at" in run


def test_list_experiment_runs_returns_run_records(tmp_path: Path) -> None:
    run_dir = tmp_path / "experiments" / "runs" / "run-a"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "run-a",
                "model_name": "movielens-popularity",
                "version": "v1",
                "artifact_uri": "artifacts/recommendation/v1",
                "metrics_uri": "artifacts/recommendation/v1/metrics.json",
                "status": "completed",
            }
        )
    )

    runs = list_experiment_runs(tmp_path / "experiments" / "runs")

    assert runs == [
        {
            "run_id": "run-a",
            "model_name": "movielens-popularity",
            "version": "v1",
            "artifact_uri": "artifacts/recommendation/v1",
            "metrics_uri": "artifacts/recommendation/v1/metrics.json",
            "status": "completed",
        }
    ]


def test_register_list_get_and_set_active_model_version(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry" / "models.json"

    version = register_model_version(
        registry_path=registry_path,
        model_name="movielens-popularity",
        version="v1",
        artifact_uri="artifacts/recommendation/v1",
        metrics_uri="artifacts/recommendation/v1/metrics.json",
        stage="candidate",
        set_active=False,
    )

    assert version["model_name"] == "movielens-popularity"
    assert version["version"] == "v1"
    assert version["stage"] == "candidate"
    assert "created_at" in version
    assert list_model_versions(registry_path) == [version]
    assert get_model_version(registry_path, "movielens-popularity", "v1") == version
    assert get_active_model(registry_path, "movielens-popularity") is None

    active = set_active_model(registry_path, "movielens-popularity", "v1")

    assert active == version
    assert get_active_model(registry_path, "movielens-popularity") == version
    assert json.loads(registry_path.read_text())["active"] == {"movielens-popularity": "v1"}


def test_train_recommender_from_config_registers_model_version(tmp_path: Path) -> None:
    config_path = tmp_path / "foundation-recommender.yaml"
    registry_path = tmp_path / "registry" / "models.json"
    config_path.write_text(
        f"""
pipeline:
  name: foundation-recommender
  version: config-registry-v1

dataset:
  ratings_path: {FIXTURE_DIR / "ratings.csv"}
  movies_path: {FIXTURE_DIR / "movies.csv"}

model:
  type: popularity
  hyperparams:
    min_rating: 4.0

artifacts:
  artifact_dir: {tmp_path / "artifacts"}

experiments:
  tracking_dir: {tmp_path / "experiments" / "runs"}
  run_id: config-registry-run

registry:
  path: {registry_path}
  stage: candidate
  set_active: true
""".strip()
    )

    result = train_recommender_from_config(config_path)
    active = get_active_model(registry_path, "movielens-popularity")

    assert active is not None
    assert active["version"] == "config-registry-v1"
    assert active["artifact_uri"] == result.uri
    assert active["metrics_uri"] == result.metrics_uri
    assert active["stage"] == "candidate"


def test_recommend_top_k_from_registry_uses_active_model(tmp_path: Path) -> None:
    result = train_popularity_recommender(
        ratings_path=FIXTURE_DIR / "ratings.csv",
        movies_path=FIXTURE_DIR / "movies.csv",
        artifact_dir=tmp_path / "artifacts",
        version="active-v1",
        min_rating=4.0,
    )
    registry_path = tmp_path / "registry" / "models.json"
    register_model_version(
        registry_path=registry_path,
        model_name="movielens-popularity",
        version="active-v1",
        artifact_uri=result.uri,
        metrics_uri=result.metrics_uri,
        stage="production",
        set_active=True,
    )

    recommendations = recommend_top_k_from_registry(registry_path, "movielens-popularity", top_k=2)

    assert [item["movieId"] for item in recommendations] == [1, 3]


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
