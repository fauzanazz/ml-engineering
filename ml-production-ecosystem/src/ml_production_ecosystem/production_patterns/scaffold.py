"""Project scaffold generator for common ML production starting points."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TEMPLATE_ROOT = PROJECT_ROOT / "templates" / "scaffold"
SUPPORTED_PRESETS = ("kaggle", "served-model", "enterprise-pipeline")
METADATA_FILE = "template.yaml"


@dataclass(frozen=True)
class ScaffoldRequest:
    preset: str
    name: str
    target: Path
    force: bool = False


@dataclass(frozen=True)
class ScaffoldResult:
    preset: str
    name: str
    package_name: str
    target: Path
    written_paths: tuple[Path, ...]


def package_name_from_project(project_name: str) -> str:
    package_name = re.sub(r"[^a-zA-Z0-9]+", "_", project_name.strip().lower()).strip("_")
    if not package_name:
        raise ValueError("Project name must contain at least one letter or number.")
    if package_name[0].isdigit():
        package_name = f"ml_{package_name}"
    return package_name


def scaffold_project(request: ScaffoldRequest) -> ScaffoldResult:
    preset = request.preset.strip().lower()
    if preset not in SUPPORTED_PRESETS:
        allowed = ", ".join(SUPPORTED_PRESETS)
        raise ValueError(f"Unsupported preset '{request.preset}'. Choose one of: {allowed}.")

    template_dir = TEMPLATE_ROOT / preset
    if not template_dir.exists():
        raise FileNotFoundError(f"Template not found: {template_dir}")

    target = request.target.expanduser().resolve()
    if target.exists() and any(target.iterdir()) and not request.force:
        raise FileExistsError(f"Target already exists and is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)

    package_name = package_name_from_project(request.name)
    variables = {
        "project_name": request.name,
        "package_name": package_name,
        "preset": preset,
    }
    written_paths = tuple(_copy_template(template_dir, target, variables))
    return ScaffoldResult(
        preset=preset,
        name=request.name,
        package_name=package_name,
        target=target,
        written_paths=written_paths,
    )


def _copy_template(template_dir: Path, target: Path, variables: dict[str, str]) -> list[Path]:
    written_paths: list[Path] = []
    for source_path in sorted(path for path in template_dir.rglob("*") if path.is_file()):
        if source_path.name == METADATA_FILE:
            continue
        relative_path = source_path.relative_to(template_dir)
        rendered_relative_path = Path(
            *(_render_text(part, variables) for part in relative_path.parts)
        )
        target_path = target / rendered_relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(_render_text(source_path.read_text(), variables))
        written_paths.append(target_path)
    return written_paths


def _render_text(value: str, variables: dict[str, str]) -> str:
    rendered = value
    for name, replacement in variables.items():
        rendered = rendered.replace(f"{{{{{name}}}}}", replacement)
    return rendered
