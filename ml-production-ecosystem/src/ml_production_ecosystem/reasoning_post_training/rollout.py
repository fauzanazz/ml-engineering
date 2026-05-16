from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .core import Rollout, TextPolicy


class RolloutGenerator(Protocol):
    def generate(self, policy: TextPolicy, prompts: list[str], candidates: int) -> list[Rollout]: ...


@dataclass(frozen=True)
class LocalRolloutGenerator:
    worker_count: int = 1

    def generate(self, policy: TextPolicy, prompts: list[str], candidates: int) -> list[Rollout]:
        if self.worker_count < 1:
            raise ValueError("worker_count must be at least 1")
        return [rollout for shard in shard_prompts(prompts, self.worker_count) for prompt in shard for rollout in policy.generate(prompt, candidates)]


@dataclass(frozen=True)
class ProviderRolloutGenerator:
    provider_name: str
    queue_name: str

    def generate(self, policy: TextPolicy, prompts: list[str], candidates: int) -> list[Rollout]:
        raise NotImplementedError("distributed provider rollout generation requires provider adapter implementation")


def shard_prompts(prompts: list[str], worker_count: int) -> list[list[str]]:
    if worker_count < 1:
        raise ValueError("worker_count must be at least 1")
    shards = [[] for _ in range(worker_count)]
    for index, prompt in enumerate(prompts):
        shards[index % worker_count].append(prompt)
    return [shard for shard in shards if shard]
