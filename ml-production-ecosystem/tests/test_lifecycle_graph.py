from pathlib import Path

from ml_production_ecosystem.production_patterns.lifecycle_graph import build_lifecycle_graph, mermaid_text


def test_mermaid_text_contains_full_lifecycle_flow() -> None:
    graph = mermaid_text()

    for node in ["Add Data", "Train Candidate", "Offline Validation", "Approve Deployment", "Drift Detection"]:
        assert node in graph
    assert "data --> train" in graph
    assert "drift --> continual" in graph


def test_build_lifecycle_graph_writes_mermaid_and_html(tmp_path: Path) -> None:
    mermaid_path = tmp_path / "lifecycle.mmd"
    html_path = tmp_path / "lifecycle.html"

    summary = build_lifecycle_graph(mermaid_path, html_path)

    assert summary["status"] == "completed"
    assert summary["format"] == "mermaid+html"
    assert summary["mermaid_path"] == str(mermaid_path)
    assert summary["html_path"] == str(html_path)
    assert "flowchart LR" in mermaid_path.read_text()
    assert "mermaid.initialize" in html_path.read_text()
    assert "dataset-manifest.json" in html_path.read_text()
