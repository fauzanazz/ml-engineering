from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "03-scale-and-reliability"
README_PATH = MODULE_PATH / "README.md"
FEATURE_DOC_PATH = ROOT / "docs" / "features" / "step-28-module-scaffold-and-scope.md"


def test_scale_and_reliability_module_scaffold_exists() -> None:
    assert MODULE_PATH.exists()
    assert README_PATH.exists()
    assert FEATURE_DOC_PATH.exists()


def test_scale_and_reliability_scope_is_defined() -> None:
    readme = README_PATH.read_text()

    for required_scope in [
        "scale",
        "reliability",
        "load behavior",
        "failure handling",
    ]:
        assert required_scope in readme


def test_scale_and_reliability_out_of_scope_is_defined() -> None:
    readme = README_PATH.read_text()

    for out_of_scope in [
        "full cloud infrastructure",
        "Kubernetes production",
        "real million traffic",
    ]:
        assert out_of_scope in readme
