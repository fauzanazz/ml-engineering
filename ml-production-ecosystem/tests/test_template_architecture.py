from pathlib import Path
import tomllib

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = ROOT / "src" / "ml_production_ecosystem"
TEMPLATE_ROOT = ROOT / "templates" / "scaffold"


def test_runtime_code_lives_under_src_package() -> None:
    for package in [
        "recommendation",
        "production_patterns",
        "scale_reliability",
        "reasoning_post_training",
        "shared",
    ]:
        assert (RUNTIME_ROOT / package / "__init__.py").exists()


def test_stage_folders_no_longer_contain_importable_runtime_packages() -> None:
    assert not (ROOT / "src" / "ml_production_ecosystem" / "legacy_foundation").exists()
    assert not (ROOT / "src" / "ml_production_ecosystem" / "legacy_production_patterns").exists()
    assert not (ROOT / "docs" / "domains" / "scale-reliability" / "scale_reliability").exists()
    assert not (ROOT / "src" / "ml_production_ecosystem" / "legacy_reasoning_post_training").exists()
    assert not (ROOT / "shared").exists()


def test_scaffold_templates_have_metadata_contracts() -> None:
    for preset in ["kaggle", "served-model", "enterprise-pipeline"]:
        metadata_path = TEMPLATE_ROOT / preset / "template.yaml"
        metadata = yaml.safe_load(metadata_path.read_text())

        assert metadata["name"] == preset
        assert metadata["version"] == 1
        assert metadata["contract"]["required_variables"] == [
            "project_name",
            "package_name",
            "preset",
        ]
        assert "generated_paths" in metadata["contract"]


def test_examples_are_not_packaged_runtime_code() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())

    assert (ROOT / "examples" / "samples" / "generic_classifier" / "train.py").exists()
    assert pyproject["tool"]["setuptools"]["packages"]["find"]["where"] == ["src"]
