from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_production_patterns_scaffold_exists() -> None:
    base = ROOT / "src" / "ml_production_ecosystem" / "production_patterns"

    expected_paths = [
        base / "__init__.py",
        base / "batch_inference.py",
        base / "quality_gate.py",
        base / "retraining.py",
        ROOT / "docs" / "domains" / "production-patterns" / "online-serving.md",
        ROOT / "docs" / "domains" / "production-patterns" / "batch-inference.md",
        ROOT / "docs" / "domains" / "production-patterns" / "retraining.md",
        ROOT / "docs" / "domains" / "production-patterns" / "monitoring-loop.md",
    ]

    for path in expected_paths:
        assert path.exists(), f"Missing production pattern scaffold path: {path}"


def test_production_patterns_readme_classifies_foundation_boundary() -> None:
    readme = (ROOT / "docs" / "domains" / "production-patterns" / "README.md").read_text()

    assert "Progress Through Step 27" in readme
    assert "01 Foundation is closed at Step 10" in readme
    assert "Step 11 batch inference is treated as transition work" in readme
    assert "online serving" in readme
    assert "batch inference" in readme
    assert "scheduled retraining" in readme
    assert "monitoring loop" in readme


def test_production_batch_wrapper_reuses_foundation_batch_entrypoint() -> None:
    wrapper = (ROOT / "src" / "ml_production_ecosystem" / "production_patterns" / "batch_inference.py").read_text()

    assert "from ml_production_ecosystem.recommendation.batch import main as foundation_batch_main" in wrapper
    assert "foundation_batch_main()" in wrapper


def test_production_patterns_cli_is_registered() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())

    assert pyproject["project"]["scripts"]["production-batch-recommend"] == "ml_production_ecosystem.production_patterns.batch_inference:main"
    assert pyproject["project"]["scripts"]["production-retrain"] == "ml_production_ecosystem.production_patterns.retraining:main"
    assert pyproject["tool"]["pytest"]["ini_options"]["pythonpath"] == ["src"]
    assert pyproject["tool"]["setuptools"]["packages"]["find"]["where"] == ["src"]
