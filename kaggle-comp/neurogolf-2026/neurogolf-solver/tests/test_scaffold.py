from pathlib import Path
from zipfile import ZipFile

from neurogolf_solver.data import list_task_paths, load_task, summarize_task
from neurogolf_solver.submission import build_baseline_submission, build_identity_model, build_submission_path, zip_models


def test_dataset_is_available() -> None:
    assert len(list_task_paths()) == 400
    assert set(load_task(1)) == {"train", "test", "arc-gen"}
    assert summarize_task(1)["train"] > 0


def test_identity_model_and_submission_zip(tmp_path: Path) -> None:
    model_path = build_identity_model(1, tmp_path)
    submission_path = zip_models([model_path], tmp_path / "submission.zip")

    assert model_path.name == "task001.onnx"
    assert build_submission_path().endswith("submission.zip")
    with ZipFile(submission_path) as archive:
        assert archive.namelist() == ["task001.onnx"]


def test_baseline_submission_contains_all_tasks() -> None:
    submission_path = build_baseline_submission(list(range(1, 401)))
    try:
        with ZipFile(submission_path) as archive:
            names = archive.namelist()
        assert len(names) == 400
        assert names[0] == "task001.onnx"
        assert names[-1] == "task400.onnx"
    finally:
        submission_path.unlink(missing_ok=True)
        output_dir = submission_path.parent / "outputs"
        for model_path in output_dir.glob("task*.onnx"):
            model_path.unlink()
