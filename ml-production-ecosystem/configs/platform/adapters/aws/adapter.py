"""Plan-backed aws provider adapter."""

from pathlib import Path

from ml_production_ecosystem.shared.platform import DeploymentExecution, InfrastructurePlan, PlatformPlanAdapter

DEFAULT_PLAN_PATH = Path("configs/platform/iac/aws/platform-plan.yaml")


class AwsProviderAdapter:
    """Thin aws adapter backed by provider-neutral platform plan."""

    provider = "aws"

    def __init__(self, plan_path: Path = DEFAULT_PLAN_PATH) -> None:
        self._adapter = PlatformPlanAdapter(plan_path)

    def plan(self, environment: str) -> InfrastructurePlan:
        """Return aws resource and secret references for environment."""

        return self._adapter.plan(environment)

    def deploy(self, environment: str, dry_run: bool = True) -> DeploymentExecution:
        """Execute or preview aws deployment without importing provider SDKs."""

        return self._adapter.deploy(environment, dry_run)
