from {{package_name}}.submission import build_submission_path


def test_submission_path() -> None:
    assert build_submission_path() == "submission.csv"
