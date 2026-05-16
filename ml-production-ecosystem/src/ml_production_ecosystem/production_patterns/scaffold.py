"""Project scaffold generator for common ML production starting points."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TEMPLATE_ROOT = PROJECT_ROOT / "templates" / "scaffold"
SUPPORTED_PRESETS = ("kaggle", "generic-classifier", "served-model", "asr-served-model", "recommendation", "batch-inference", "existing-model-wrapper", "llm-post-training", "enterprise-pipeline")
METADATA_FILE = "template.yaml"
SUPPORTED_TASKS = ("classification", "regression", "recommendation", "speech-to-text", "nlp", "computer-vision", "forecasting", "llm-post-training", "batch-inference", "existing-model")
SUPPORTED_MODEL_TYPES = ("sklearn", "xgboost", "pytorch", "transformers", "whisper", "llm", "rules", "external")
SUPPORTED_BACKENDS = ("local", "fastapi", "batch", "spark", "airflow", "kubernetes", "serverless", "external-command")
SUPPORTED_INFRA = ("api", "batch", "registry", "quality-gate", "monitoring", "drift", "retraining", "rollback", "docker", "kubernetes", "secrets", "ci")
PRESET_DEFAULTS = {
    "kaggle": ("classification", "sklearn", "batch", ("batch", "quality-gate", "ci")),
    "generic-classifier": ("classification", "sklearn", "batch", ("quality-gate", "registry", "ci")),
    "served-model": ("classification", "sklearn", "fastapi", ("api", "quality-gate", "monitoring", "docker", "ci")),
    "asr-served-model": ("speech-to-text", "whisper", "fastapi", ("api", "quality-gate", "monitoring", "registry", "ci")),
    "recommendation": ("recommendation", "sklearn", "batch", ("batch", "quality-gate", "registry", "monitoring", "ci")),
    "batch-inference": ("batch-inference", "external", "batch", ("batch", "monitoring", "ci")),
    "existing-model-wrapper": ("existing-model", "external", "external-command", ("quality-gate", "registry", "monitoring", "ci")),
    "llm-post-training": ("llm-post-training", "llm", "local", ("quality-gate", "registry", "monitoring", "ci")),
    "enterprise-pipeline": ("classification", "external", "airflow", ("batch", "registry", "quality-gate", "monitoring", "drift", "retraining", "rollback", "docker", "kubernetes", "secrets", "ci")),
}


@dataclass(frozen=True)
class ScaffoldRequest:
    preset: str
    name: str
    target: Path
    force: bool = False
    task: str | None = None
    model_type: str | None = None
    backend: str | None = None
    infra: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScaffoldResult:
    preset: str
    name: str
    package_name: str
    target: Path
    written_paths: tuple[Path, ...]
    task: str
    model_type: str
    backend: str
    infra: tuple[str, ...]


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

    task, model_type, backend, infra = resolve_scaffold_axes(request)
    package_name = package_name_from_project(request.name)
    variables = {
        "project_name": request.name,
        "package_name": package_name,
        "preset": preset,
        "task": task,
        "model_type": model_type,
        "backend": backend,
        "infra": ", ".join(infra),
    }
    written_paths = _copy_template(template_dir, target, variables)
    written_paths.extend(_write_modular_files(target, request.name, preset, task, model_type, backend, infra))
    return ScaffoldResult(
        preset=preset,
        name=request.name,
        package_name=package_name,
        target=target,
        written_paths=tuple(written_paths),
        task=task,
        model_type=model_type,
        backend=backend,
        infra=infra,
    )


def resolve_scaffold_axes(request: ScaffoldRequest) -> tuple[str, str, str, tuple[str, ...]]:
    default_task, default_model_type, default_backend, default_infra = PRESET_DEFAULTS[request.preset.strip().lower()]
    task = request.task or default_task
    model_type = request.model_type or default_model_type
    backend = request.backend or default_backend
    infra = request.infra or default_infra
    _validate_axis("task", task, SUPPORTED_TASKS)
    _validate_axis("model type", model_type, SUPPORTED_MODEL_TYPES)
    _validate_axis("backend", backend, SUPPORTED_BACKENDS)
    for item in infra:
        _validate_axis("infra", item, SUPPORTED_INFRA)
    return task, model_type, backend, tuple(dict.fromkeys(infra))

def _validate_axis(label: str, value: str, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ValueError(f"Unsupported {label} '{value}'. Choose one of: {', '.join(allowed)}.")

def _write_modular_files(
    target: Path,
    project_name: str,
    preset: str,
    task: str,
    model_type: str,
    backend: str,
    infra: tuple[str, ...],
) -> list[Path]:
    config_path = target / "ml-struct.yaml"
    checklist_path = target / "docs" / "infra-checklist.md"
    config_path.write_text(
        "project: " + project_name + "\n"
        "preset: " + preset + "\n"
        "task: " + task + "\n"
        "model_type: " + model_type + "\n"
        "backend: " + backend + "\n"
        "infra:\n" + "".join(f"  - {item}\n" for item in infra)
    )
    checklist_path.parent.mkdir(parents=True, exist_ok=True)
    checklist_path.write_text(
        "# Infra Checklist\n\n"
        f"Task: `{task}`\n"
        f"Model type: `{model_type}`\n"
        f"Backend: `{backend}`\n\n"
        + "".join(f"- [ ] {item}\n" for item in infra)
    )
    return [config_path, checklist_path]

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
