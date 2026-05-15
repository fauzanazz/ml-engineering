from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .core import LocalPatternPolicy, TextPolicy


class Backend(Protocol):
    name: str
    def create_policy(self, seed: int) -> TextPolicy: ...


@dataclass(frozen=True)
class LocalBackend:
    name: str = "local"

    def create_policy(self, seed: int) -> TextPolicy:
        return LocalPatternPolicy(seed=seed)


@dataclass(frozen=True)
class ProviderBackend:
    name: str
    endpoint: str

    def create_policy(self, seed: int) -> TextPolicy:
        raise NotImplementedError("provider backend boundary configured, but remote training is intentionally adapter-only")


def backend_from_config(config: dict[str, object]) -> Backend:
    backend_name = str(config.get("backend", "local"))
    if backend_name == "local":
        return LocalBackend()
    return ProviderBackend(name=backend_name, endpoint=str(config.get("provider_endpoint", "")))
