from pathlib import Path
import json

from {{package_name}}.evaluate import run_evaluation


def test_evaluation_writes_report(tmp_path: Path) -> None:
    metrics_path = tmp_path / "reports" / "metrics.json"
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text(json.dumps({"accuracy": 1.0}))
    report_path = tmp_path / "artifacts" / "reports" / "evaluation.json"

    report = run_evaluation(metrics_path, report_path)

    assert report["status"] == "completed"
    assert report_path.exists()
    assert report["metrics"]["accuracy"] == 1.0
