#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORK_DIR=$(mktemp -d)
SERVER_PID=""

stop_tree() {
  local pid=$1
  local child
  while read -r child; do
    [[ -n "$child" ]] && stop_tree "$child"
  done < <(pgrep -P "$pid" 2>/dev/null || true)
  kill -TERM "$pid" 2>/dev/null || true
}

cleanup() {
  if [[ -n "$SERVER_PID" ]]; then
    stop_tree "$SERVER_PID"
  fi
  rm -rf "$WORK_DIR"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$ROOT_DIR"
rm -rf dist
uv build

WHEEL=$(printf '%s\n' dist/*.whl)
SDIST=$(printf '%s\n' dist/*.tar.gz)
python3 - "$ROOT_DIR/src/ml_production_ecosystem/templates/scaffold" "$WHEEL" "$SDIST" <<'PY'
from pathlib import Path
import sys
import tarfile
import zipfile

template_root = Path(sys.argv[1])
wheel_path = Path(sys.argv[2])
sdist_path = Path(sys.argv[3])
expected = {
    path.relative_to(template_root).as_posix()
    for path in template_root.rglob("*")
    if path.is_file()
}
wheel_prefix = "ml_production_ecosystem/templates/scaffold/"
sdist_prefix = "ml_production_ecosystem-0.1.0/src/ml_production_ecosystem/templates/scaffold/"

with zipfile.ZipFile(wheel_path) as archive:
    wheel_files = {
        name.removeprefix(wheel_prefix)
        for name in archive.namelist()
        if name.startswith(wheel_prefix) and not name.endswith("/")
    }
with tarfile.open(sdist_path) as archive:
    sdist_files = {
        member.name.removeprefix(sdist_prefix)
        for member in archive.getmembers()
        if member.isfile() and member.name.startswith(sdist_prefix)
    }

assert wheel_files == expected, (expected - wheel_files, wheel_files - expected)
assert sdist_files == expected, (expected - sdist_files, sdist_files - expected)
PY

uv venv --python 3.13 "$WORK_DIR/.venv"
VENV_PYTHON="$WORK_DIR/.venv/bin/python"
uv pip install --python "$VENV_PYTHON" "$WHEEL"

cd "$WORK_DIR"
"$WORK_DIR/.venv/bin/create-ml-struct" churn-api --preset served-model --no-input
cd churn-api
uv run pytest
uv run python -m churn_api.train

python3 - <<'PY'
import json
from pathlib import Path

summary = json.loads(Path("artifacts/reports/training-summary.json").read_text())
metrics = json.loads(Path("artifacts/reports/metrics.json").read_text())
assert summary["model_name"] == "churn_api"
assert metrics["accuracy"] == 1.0
PY

PORT=18080 uv run serve >"$WORK_DIR/server.log" 2>&1 &
SERVER_PID=$!
for _ in {1..30}; do
  if curl -fsS --max-time 2 http://127.0.0.1:18080/health >"$WORK_DIR/health.json"; then
    break
  fi
  sleep 1
done
curl -fsS --max-time 2 http://127.0.0.1:18080/health >"$WORK_DIR/health.json"
curl -fsS --max-time 2 \
  -H 'content-type: application/json' \
  -d '{"features":{"a":1.0,"b":2.0}}' \
  http://127.0.0.1:18080/predict >"$WORK_DIR/predict.json"

python3 - "$WORK_DIR/health.json" "$WORK_DIR/predict.json" <<'PY'
import json
from pathlib import Path
import sys

health = json.loads(Path(sys.argv[1]).read_text())
prediction = json.loads(Path(sys.argv[2]).read_text())
expected_identity = {"model_name": "churn_api-dummy", "model_version": "0.1.0"}
assert health == {"status": "ok", **expected_identity}
assert prediction == {"prediction": True, **expected_identity}
PY

stop_tree "$SERVER_PID"
SERVER_PID=""
docker build -t churn-api:e2e .
