#!/usr/bin/env bash
set -euo pipefail

uv run production-apply-local-platform \
  --output-path artifacts/reports/production-patterns/local-platform-apply.json

uv run production-validate-local-secret-injections \
  --output-path artifacts/reports/production-patterns/local-secret-injections.json

uv run production-validate-local-kubernetes \
  --output-path artifacts/reports/production-patterns/local-kubernetes-validation.json

uv run production-validate-local-scheduler \
  --output-path artifacts/reports/production-patterns/local-scheduler-validation.json

uv run production-run-local-scheduler \
  --job-name lifecycle-status \
  --output-path artifacts/reports/production-patterns/local-scheduler-run.json

uv run production-lifecycle-demo \
  --config configs/local-lifecycle-demo.yaml \
  --output-path artifacts/reports/production-patterns/local-lifecycle-demo.json \
  --graph-path artifacts/reports/production-patterns/local-lifecycle-demo.mmd \
  --graph-html-path artifacts/reports/production-patterns/local-lifecycle-demo.html

uv run production-lifecycle-status \
  --output-path artifacts/reports/production-patterns/local-lifecycle-status.json

python - <<'PY'
import json
from pathlib import Path

summary = json.loads(Path("artifacts/reports/production-patterns/local-lifecycle-demo.json").read_text())
required = {
    "platform": "passed",
    "model_contract": "ready",
    "dataset": "ready",
    "offline_validation": "passed",
}
for section, expected in required.items():
    actual = summary[section]["status"]
    if actual != expected:
        raise SystemExit(f"{section} status {actual!r}, expected {expected!r}")

if summary["training"]["status"] != "completed":
    raise SystemExit("training did not complete")
if summary["graph"]["status"] != "completed":
    raise SystemExit("graph did not complete")

status = json.loads(Path("artifacts/reports/production-patterns/local-lifecycle-status.json").read_text())
if status["status"] != "incomplete":
    raise SystemExit(f"status {status['status']!r}, expected 'incomplete' without deployment reports")

scheduler = json.loads(Path("artifacts/reports/production-patterns/local-scheduler-run.json").read_text())
if scheduler["status"] != "passed":
    raise SystemExit(f"scheduler status {scheduler['status']!r}, expected 'passed'")
if scheduler["mode"] != "local-scheduler-runtime":
    raise SystemExit(f"scheduler mode {scheduler['mode']!r}, expected 'local-scheduler-runtime'")

print("local lifecycle smoke passed")
PY
