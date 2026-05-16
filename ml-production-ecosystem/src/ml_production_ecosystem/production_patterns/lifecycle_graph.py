"""Lifecycle graph artifacts for local ML production flow."""

from pathlib import Path
import argparse
import html
import json

DEFAULT_MERMAID_PATH = Path("02-production-patterns/reports/lifecycle-demo.mmd")
DEFAULT_HTML_PATH = Path("02-production-patterns/reports/lifecycle-demo.html")

NODES = (
    ("contract", "Validate Model Contract", "02-production-patterns/reports/model-contract-manifest.json"),
    ("data", "Add Data", "02-production-patterns/reports/dataset-manifest.json"),
    ("train", "Train Candidate", "01-foundation/experiments/runs"),
    ("validate", "Offline Validation", "02-production-patterns/reports/offline-validation.json"),
    ("approve", "Approve Deployment", "02-production-patterns/reports/approval-decision.json"),
    ("demo", "Deployment Demo Test", "02-production-patterns/reports/deployment-demo.json"),
    ("drift", "Drift Detection", "02-production-patterns/reports/drift-report.json"),
    ("continual", "Continual Learning", "02-production-patterns/reports/continual-learning-decision.json"),
)

EDGES = (
    ("contract", "data"),
    ("data", "train"),
    ("train", "validate"),
    ("validate", "approve"),
    ("approve", "demo"),
    ("demo", "drift"),
    ("drift", "continual"),
)


def mermaid_text() -> str:
    lines = ["flowchart LR"]
    lines.extend(f'    {node_id}["{label}"]' for node_id, label, _ in NODES)
    lines.extend(f"    {source} --> {target}" for source, target in EDGES)
    lines.extend(["    validate -->|fail| train", "    drift -->|breach| train", "    demo -->|fail| approve", ""])
    return "\n".join(lines)


def _html_text(mermaid: str) -> str:
    links = "\n".join(
        f'<li><a href="../../{html.escape(path)}">{html.escape(label)}</a></li>' for _, label, path in NODES
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Local ML Lifecycle Flow</title>
  <script type="module">import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs"; mermaid.initialize({{startOnLoad: true}});</script>
</head>
<body>
  <h1>Local ML Lifecycle Flow</h1>
  <pre class="mermaid">
{html.escape(mermaid)}
  </pre>
  <h2>Artifacts</h2>
  <ul>
{links}
  </ul>
</body>
</html>
"""


def build_lifecycle_graph(
    mermaid_path: Path = DEFAULT_MERMAID_PATH,
    html_path: Path = DEFAULT_HTML_PATH,
) -> dict[str, object]:
    mermaid = mermaid_text()
    mermaid_path.parent.mkdir(parents=True, exist_ok=True)
    mermaid_path.write_text(mermaid)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(_html_text(mermaid))
    return {
        "status": "completed",
        "format": "mermaid+html",
        "mermaid_path": str(mermaid_path),
        "html_path": str(html_path),
        "nodes": [{"id": node_id, "label": label, "artifact": path} for node_id, label, path in NODES],
        "edges": [{"source": source, "target": target} for source, target in EDGES],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Write lifecycle graph Mermaid and HTML artifacts.")
    parser.add_argument("--mermaid-path", type=Path, default=DEFAULT_MERMAID_PATH)
    parser.add_argument("--html-path", type=Path, default=DEFAULT_HTML_PATH)
    args = parser.parse_args()

    summary = build_lifecycle_graph(args.mermaid_path, args.html_path)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
