from __future__ import annotations

import json
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import onnxruntime as ort

CHANNELS = 10
HEIGHT = 30
WIDTH = 30
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT.parent / "data"
SOURCE_DIR = PROJECT_ROOT / "hf_submission"
VALID_DIR = PROJECT_ROOT / "public_valid_submission"
SUBMISSION_PATH = PROJECT_ROOT / "submission.zip"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LABEL = "hf-enhanced-public-valid"


def to_onehot(grid: list[list[int]]) -> np.ndarray:
    array = np.zeros((1, CHANNELS, HEIGHT, WIDTH), dtype=np.float32)
    for row_index, row in enumerate(grid):
        for col_index, color in enumerate(row):
            array[0, color, row_index, col_index] = 1.0
    return array


def passes_all_examples(model_path: Path, task_path: Path) -> bool:
    task = json.loads(task_path.read_text())
    try:
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    except Exception:
        return False
    for split_name in ["train", "test", "arc-gen"]:
        for example in task[split_name]:
            expected = to_onehot(example["output"])
            try:
                actual = session.run(["output"], {"input": to_onehot(example["input"])})[0]
            except Exception:
                return False
            if not np.array_equal((actual > 0.0).astype(np.float32), expected):
                return False
    return True


def main() -> None:
    if VALID_DIR.exists():
        shutil.rmtree(VALID_DIR)
    VALID_DIR.mkdir(parents=True)
    passed: list[Path] = []
    failed: list[str] = []
    for model_path in sorted(SOURCE_DIR.glob("task*.onnx")):
        task_path = DATA_DIR / f"{model_path.stem}.json"
        if passes_all_examples(model_path, task_path):
            destination = VALID_DIR / model_path.name
            shutil.copy2(model_path, destination)
            passed.append(destination)
        else:
            failed.append(model_path.name)
        print(f"checked={len(passed)+len(failed)} passed={len(passed)} failed={len(failed)}", end="\r")
    print()

    with ZipFile(SUBMISSION_PATH, "w", ZIP_DEFLATED) as archive:
        for model_path in passed:
            archive.write(model_path, arcname=model_path.name)

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    artifact_zip = ARTIFACTS_DIR / f"{LABEL}.zip"
    shutil.copy2(SUBMISSION_PATH, artifact_zip)
    manifest = {
        "label": LABEL,
        "strategy": "HF enhanced analytical + least-squares conv; filtered by train/test/arc-gen public examples",
        "task_count": len(passed),
        "failed_count": len(failed),
        "first_task": passed[0].name if passed else None,
        "last_task": passed[-1].name if passed else None,
        "submission_path": str(SUBMISSION_PATH),
        "artifact_zip": str(artifact_zip),
        "submission_bytes": SUBMISSION_PATH.stat().st_size,
    }
    (ARTIFACTS_DIR / f"{LABEL}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
