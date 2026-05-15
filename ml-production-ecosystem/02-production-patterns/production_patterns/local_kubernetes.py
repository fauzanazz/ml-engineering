"""Validate local Kubernetes manifests without requiring a cluster."""

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

DEFAULT_MANIFEST_PATH = Path("04-platform-and-cloud/iac/local/kubernetes/foundation-api.yaml")
DEFAULT_OUTPUT_PATH = Path("02-production-patterns/reports/local-kubernetes-validation.json")
FORBIDDEN_KINDS = {"Secret"}


def validate_local_kubernetes(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    documents = [doc for doc in yaml.safe_load_all(manifest_path.read_text()) if isinstance(doc, dict)]
    kinds = [str(doc.get("kind", "")) for doc in documents]
    failures = []

    required_kinds = {"ConfigMap", "Deployment", "Service"}
    missing_kinds = sorted(required_kinds - set(kinds))
    if missing_kinds:
        failures.append(f"missing kinds: {', '.join(missing_kinds)}")
    forbidden = sorted(FORBIDDEN_KINDS & set(kinds))
    if forbidden:
        failures.append(f"forbidden committed kinds: {', '.join(forbidden)}")

    deployment = _first_kind(documents, "Deployment")
    service = _first_kind(documents, "Service")
    if deployment:
        failures.extend(_deployment_failures(deployment))
    if service:
        failures.extend(_service_failures(service))

    report: dict[str, Any] = {
        "status": "passed" if not failures else "failed",
        "manifest_path": str(manifest_path),
        "kinds": kinds,
        "failures": failures,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _deployment_failures(deployment: dict[str, Any]) -> list[str]:
    failures = []
    spec = deployment.get("spec", {})
    template = spec.get("template", {}) if isinstance(spec, dict) else {}
    pod_spec = template.get("spec", {}) if isinstance(template, dict) else {}
    containers = pod_spec.get("containers", []) if isinstance(pod_spec, dict) else []
    if not containers:
        return ["deployment has no containers"]
    container = containers[0]
    if not str(container.get("image", "")).endswith(":local"):
        failures.append("deployment image must use local tag")
    if "foundation-serve-recommender" not in " ".join(str(part) for part in container.get("command", [])):
        failures.append("deployment command must start foundation serving")
    env = container.get("env", [])
    if _has_literal_secret_value(env):
        failures.append("deployment env must not contain literal secret values")
    if not _has_secret_key_ref(env, "LOCAL_MODEL_REGISTRY_TOKEN"):
        failures.append("deployment must reference LOCAL_MODEL_REGISTRY_TOKEN via secretKeyRef")
    if "readinessProbe" not in container:
        failures.append("deployment missing readinessProbe")
    if "livenessProbe" not in container:
        failures.append("deployment missing livenessProbe")
    return failures


def _service_failures(service: dict[str, Any]) -> list[str]:
    spec = service.get("spec", {})
    if not isinstance(spec, dict):
        return ["service spec must be object"]
    ports = spec.get("ports", [])
    if not ports or not isinstance(ports, list):
        return ["service must expose ports"]
    if int(ports[0].get("port", 0)) != 8000:
        return ["service must expose port 8000"]
    return []


def _first_kind(documents: list[dict[str, Any]], kind: str) -> dict[str, Any] | None:
    for document in documents:
        if document.get("kind") == kind:
            return document
    return None


def _has_literal_secret_value(env: object) -> bool:
    if not isinstance(env, list):
        return False
    return any(isinstance(item, dict) and item.get("value") for item in env)


def _has_secret_key_ref(env: object, name: str) -> bool:
    if not isinstance(env, list):
        return False
    for item in env:
        if not isinstance(item, dict) or item.get("name") != name:
            continue
        value_from = item.get("valueFrom", {})
        return isinstance(value_from, dict) and "secretKeyRef" in value_from
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate local Kubernetes manifests without applying them.")
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    report = validate_local_kubernetes(args.manifest_path, args.output_path)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
