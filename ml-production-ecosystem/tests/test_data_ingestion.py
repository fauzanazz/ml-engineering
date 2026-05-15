from pathlib import Path
import json

from production_patterns.data_ingestion import build_dataset_manifest


def test_build_dataset_manifest_accepts_existing_sources(tmp_path: Path) -> None:
    ratings_path = tmp_path / "ratings.csv"
    movies_path = tmp_path / "movies.csv"
    ratings_path.write_text("userId,movieId,rating,timestamp\n1,1,5,1\n")
    movies_path.write_text("movieId,title,genres\n1,Toy Story,Adventure\n")
    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "dataset-manifest.json"
    config_path.write_text(
        f"""
pipeline:
  name: demo-dataset
dataset:
  version: v1
  schema_uri: schemas/demo/input.json
  ratings_path: {ratings_path}
  movies_path: {movies_path}
""".strip()
    )

    manifest = build_dataset_manifest(config_path, output_path)

    assert manifest["status"] == "ready"
    assert manifest["name"] == "demo-dataset"
    assert manifest["version"] == "v1"
    assert manifest["schema_uri"] == "schemas/demo/input.json"
    assert manifest["missing_sources"] == []
    assert json.loads(output_path.read_text()) == manifest


def test_build_dataset_manifest_blocks_missing_sources(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "dataset-manifest.json"
    missing_path = tmp_path / "missing.csv"
    config_path.write_text(
        f"""
dataset:
  name: demo
  ratings_path: {missing_path}
""".strip()
    )

    manifest = build_dataset_manifest(config_path, output_path)

    assert manifest["status"] == "blocked"
    assert manifest["missing_sources"] == [str(missing_path)]
