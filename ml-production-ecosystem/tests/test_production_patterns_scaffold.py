from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_production_patterns_scaffold_exists() -> None:
    base = ROOT / "02-production-patterns"

    expected_paths = [
        base / "production_patterns" / "__init__.py",
        base / "production_patterns" / "batch_inference.py",
        base / "production_patterns" / "quality_gate.py",
        base / "production_patterns" / "retraining.py",
        base / "docs" / "online-serving.md",
        base / "docs" / "batch-inference.md",
        base / "docs" / "retraining.md",
        base / "docs" / "monitoring-loop.md",
    ]

    for path in expected_paths:
        assert path.exists(), f"Missing production pattern scaffold path: {path}"


def test_production_patterns_readme_classifies_foundation_boundary() -> None:
    readme = (ROOT / "02-production-patterns" / "README.md").read_text()

    assert "Progress Through Step 26" in readme
    assert "01 Foundation is closed at Step 10" in readme
    assert "Step 11 batch inference is treated as transition work" in readme
    assert "online serving" in readme
    assert "batch inference" in readme
    assert "scheduled retraining" in readme
    assert "monitoring loop" in readme


def test_production_batch_wrapper_reuses_foundation_batch_entrypoint() -> None:
    wrapper = (ROOT / "02-production-patterns" / "production_patterns" / "batch_inference.py").read_text()

    assert "from recommendation.batch import main as foundation_batch_main" in wrapper
    assert "foundation_batch_main()" in wrapper


def test_production_patterns_cli_is_registered() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())

    assert pyproject["project"]["scripts"]["production-batch-recommend"] == "production_patterns.batch_inference:main"
    assert pyproject["project"]["scripts"]["production-retrain"] == "production_patterns.retraining:main"
    assert "02-production-patterns" in pyproject["tool"]["pytest"]["ini_options"]["pythonpath"]
    assert "production_patterns" in pyproject["tool"]["setuptools"]["packages"]
