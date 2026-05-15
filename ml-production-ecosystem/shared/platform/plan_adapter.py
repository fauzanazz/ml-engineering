"""Provider adapter backed by platform-plan YAML references."""

from pathlib import Path
from typing import Any

import yaml

from .contracts import CloudResourceRef, DeploymentAction, DeploymentExecution, InfrastructurePlan, SecretRef

class PlatformPlanAdapter:
    """Load provider-neutral infrastructure plan from YAML."""

    def __init__(self, plan_path: Path) -> None:
        self.plan_path = plan_path
        self._plan_data = _read_plan(plan_path)
        self.provider = str(self._plan_data.get("provider", ""))

    def plan(self, environment: str) -> InfrastructurePlan:
        plan_environment = str(self._plan_data.get("environment", ""))
        if plan_environment != environment:
            raise ValueError(f"plan environment {plan_environment!r} does not match {environment!r}")
        resources = tuple(
            CloudResourceRef(
                provider=self.provider,
                kind=str(resource["kind"]),
                name=str(resource["name"]),
                uri=str(resource["uri"]),
            )
            for resource in _items(self._plan_data.get("resources", []))
        )
        secrets = tuple(
            SecretRef(
                provider=str(secret["provider"]),
                name=str(secret["name"]),
                injection_target=str(secret["injection_target"]),
                policy_ref=str(secret["policy_ref"]),
            )
            for secret in _items(self._plan_data.get("secrets", []))
        )
        return InfrastructurePlan(provider=self.provider, resources=resources, secrets=secrets)

    def deploy(self, environment: str, dry_run: bool = True) -> DeploymentExecution:
        plan = self.plan(environment)
        action_status = "planned" if dry_run else "applied"
        actions = tuple(
            DeploymentAction(
                provider=self.provider,
                name=resource.name,
                kind=resource.kind,
                uri=resource.uri,
                status=action_status,
            )
            for resource in plan.resources
        )
        return DeploymentExecution(
            provider=self.provider,
            environment=environment,
            dry_run=dry_run,
            status=action_status,
            actions=actions,
        )

def _items(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]

def _read_plan(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if isinstance(data, dict):
        return data
    return {}
