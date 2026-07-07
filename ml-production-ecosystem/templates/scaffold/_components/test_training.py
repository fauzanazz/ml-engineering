from pathlib import Path
import json

from {{package_name}}.train import write_training_summary


def test_training_writes_summary_and_metrics(tmp_path: Path) -> None:
    summary_path = tmp_path / "artifacts" / "reports" / "training-summary.json"

    summary = write_training_summary(summary_path)

    assert summary["model_name"] == "{{package_name}}"
    assert summary_path.exists()
    assert json.loads((summary_path.parent / "metrics.json").read_text())
