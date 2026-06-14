from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from neurogolf_solver.models import identity_network, save_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_SUBMISSION_PATH = PROJECT_ROOT / "submission.zip"
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def task_model_name(task_num: int) -> str:
    return f"task{task_num:03d}.onnx"


def build_identity_model(task_num: int, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    return save_model(identity_network(), output_dir / task_model_name(task_num))


def build_submission_path() -> str:
    return str(DEFAULT_SUBMISSION_PATH)


def zip_models(model_paths: list[Path], submission_path: Path = DEFAULT_SUBMISSION_PATH) -> Path:
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(submission_path, "w", ZIP_DEFLATED) as archive:
        for model_path in model_paths:
            archive.write(model_path, arcname=model_path.name)
    return submission_path


def write_artifact_manifest(
    label: str,
    model_paths: list[Path],
    submission_path: Path,
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
) -> Path:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = artifacts_dir / f"{label}.json"
    manifest = {
        "label": label,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "strategy": "identity_baseline",
        "task_count": len(model_paths),
        "first_task": model_paths[0].name if model_paths else None,
        "last_task": model_paths[-1].name if model_paths else None,
        "submission_path": str(submission_path),
        "submission_bytes": submission_path.stat().st_size,
        "notes": "Baseline identity ONNX for every task; not expected to score well.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def copy_labeled_submission(label: str, submission_path: Path, artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR) -> Path:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    labeled_path = artifacts_dir / f"{label}.zip"
    labeled_path.write_bytes(submission_path.read_bytes())
    return labeled_path


def build_baseline_submission(task_nums: list[int] | None = None, label: str = "identity-baseline-400") -> Path:
    selected_task_nums = task_nums or list(range(1, 401))
    model_paths = [build_identity_model(task_num) for task_num in selected_task_nums]
    submission_path = zip_models(model_paths)
    copy_labeled_submission(label, submission_path)
    write_artifact_manifest(label, model_paths, submission_path)
    return submission_path


def build_smoke_submission(task_nums: list[int] | None = None) -> Path:
    return build_baseline_submission(task_nums or [1])


if __name__ == "__main__":
    print(build_baseline_submission())
