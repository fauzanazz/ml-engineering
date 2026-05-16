from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_foundation_docs_are_centralized() -> None:
    expected_docs = [
        ROOT / "docs" / "features" / "step-1-foundation-scaffold-and-shared-architecture.md",
        ROOT / "docs" / "mlops-tools-map.md",
        ROOT / "docs" / "run-log.md",
        ROOT / "docs" / "learning-roadmap.md",
    ]

    for path in expected_docs:
        assert path.exists(), f"Missing central doc: {path}"


def test_foundation_folder_was_removed() -> None:
    assert not (ROOT / "legacy_foundation").exists()


def test_foundation_has_local_docker_compose_but_no_cloud_ci_cd_files() -> None:
    assert (ROOT / "docker-compose.yml").exists()

    forbidden = [
        ROOT / ".github",
        ROOT / "k8s",
        ROOT / "kubernetes",
        ROOT / "terraform",
        ROOT / "helm",
    ]

    for path in forbidden:
        assert not path.exists(), f"Cloud/CI/CD file should not exist yet: {path}"
