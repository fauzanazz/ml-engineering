#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:8000}"

python - "$BASE_URL" <<'PY'
import json
import sys
import time
import urllib.error
import urllib.request

base_url = sys.argv[1].rstrip("/")


def request_json(path: str, *, method: str = "GET", body: dict[str, object] | None = None) -> dict[str, object]:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            if response.status >= 400:
                raise RuntimeError(f"{path} returned HTTP {response.status}")
            payload = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"{path} returned HTTP {error.code}: {error.read().decode('utf-8')}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"{path} request failed: {error.reason}") from error

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{path} did not return JSON: {payload[:200]}") from error
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{path} returned non-object JSON")
    return parsed


def wait_for_json(path: str, attempts: int = 20) -> dict[str, object]:
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            return request_json(path)
        except Exception as error:
            last_error = error
            time.sleep(1)
    raise RuntimeError(f"{path} did not become ready after {attempts} attempts: {last_error}")


health = wait_for_json("/health")
if health.get("status") != "ok":
    raise RuntimeError(f"/health status not ok: {health}")

metrics = request_json("/metrics.json")
for field in ["prediction_request_count", "prediction_error_count", "prediction_latency_ms_last"]:
    if field not in metrics:
        raise RuntimeError(f"/metrics.json missing field {field}: {metrics}")

drift = request_json("/drift")
for field in ["model_name", "version", "drift_score"]:
    if field not in drift:
        raise RuntimeError(f"/drift missing field {field}: {drift}")

prediction = request_json("/predict/v1", method="POST", body={"top_k": 3})
recommendations = prediction.get("recommendations")
if not isinstance(recommendations, list) or not recommendations:
    raise RuntimeError(f"/predict/v1 missing recommendations: {prediction}")

print(f"foundation-api smoke test passed: {base_url}")
PY
