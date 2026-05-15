from __future__ import annotations

from dataclasses import dataclass
import json
import random
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class ReasoningExample:
    prompt: str
    reasoning: str
    answer: str


@dataclass(frozen=True)
class Rollout:
    prompt: str
    reasoning: str
    answer: str
    reward: float = 0.0


class TextPolicy(Protocol):
    def generate(self, prompt: str, candidates: int = 1) -> list[Rollout]: ...
    def update(self, examples: list[ReasoningExample], learning_rate: float) -> None: ...


class RewardModel(Protocol):
    def score(self, prompt: str, reasoning: str, answer: str) -> float: ...


class LocalPatternPolicy:
    def __init__(self, templates: dict[str, ReasoningExample] | None = None, seed: int = 7) -> None:
        self.templates = templates or {}
        self.random = random.Random(seed)

    def generate(self, prompt: str, candidates: int = 1) -> list[Rollout]:
        if candidates < 1:
            raise ValueError("candidates must be at least 1")
        example = self.templates.get(prompt)
        base_answer = example.answer if example else _infer_arithmetic_answer(prompt)
        base_reasoning = example.reasoning if example else f"Parse problem, compute result, answer {base_answer}."
        return [self._candidate(prompt, base_reasoning, base_answer, index) for index in range(candidates)]

    def update(self, examples: list[ReasoningExample], learning_rate: float) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        for example in examples:
            self.templates[example.prompt] = example

    def _candidate(self, prompt: str, reasoning: str, answer: str, index: int) -> Rollout:
        if index == 0:
            return Rollout(prompt=prompt, reasoning=reasoning, answer=answer)
        maybe_wrong = str(int(answer) + index) if answer.lstrip("-").isdigit() else f"{answer}-{index}"
        if self.random.random() < 0.5:
            maybe_wrong = answer
        return Rollout(prompt=prompt, reasoning=f"Candidate {index}: {reasoning}", answer=maybe_wrong)


class RuleVerifierReward:
    def __init__(self, references: dict[str, str]) -> None:
        self.references = references

    def score(self, prompt: str, reasoning: str, answer: str) -> float:
        reference = self.references.get(prompt)
        if reference is None:
            return 0.0
        answer_score = 1.0 if answer.strip() == reference.strip() else -1.0
        reasoning_bonus = 0.2 if any(token in reasoning.lower() for token in ("compute", "therefore", "answer")) else 0.0
        return answer_score + reasoning_bonus


def load_examples(path: Path) -> list[ReasoningExample]:
    records = json.loads(path.read_text())
    return [ReasoningExample(**record) for record in records]


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _infer_arithmetic_answer(prompt: str) -> str:
    numbers = [int(part) for part in prompt.replace("?", "").split() if part.lstrip("-").isdigit()]
    if "minus" in prompt or "-" in prompt:
        return str(numbers[0] - sum(numbers[1:])) if numbers else "unknown"
    return str(sum(numbers)) if numbers else "unknown"
