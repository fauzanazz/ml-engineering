from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import time
import uuid

from .core import LocalPatternPolicy, ReasoningExample, Rollout, RuleVerifierReward, load_examples, save_json
from .rollout import LocalRolloutGenerator


def train_sft(examples: list[ReasoningExample], seed: int = 7) -> LocalPatternPolicy:
    policy = LocalPatternPolicy(seed=seed)
    policy.update(examples, learning_rate=1.0)
    return policy


def train_verifier(examples: list[ReasoningExample]) -> RuleVerifierReward:
    return RuleVerifierReward({example.prompt: example.answer for example in examples})


def generate_rollouts(policy: LocalPatternPolicy, prompts: list[str], candidates: int, worker_count: int = 1) -> list[Rollout]:
    return LocalRolloutGenerator(worker_count=worker_count).generate(policy, prompts, candidates)


def run_grpo_step(policy: LocalPatternPolicy, verifier: RuleVerifierReward, prompts: list[str], candidates: int) -> dict[str, object]:
    rollouts = generate_rollouts(policy, prompts, candidates)
    scored = [Rollout(item.prompt, item.reasoning, item.answer, verifier.score(item.prompt, item.reasoning, item.answer)) for item in rollouts]
    by_prompt = {prompt: [item for item in scored if item.prompt == prompt] for prompt in prompts}
    winners = [max(items, key=lambda item: item.reward) for items in by_prompt.values()]
    policy.update([ReasoningExample(item.prompt, item.reasoning, item.answer) for item in winners if item.reward > 0], learning_rate=0.2)
    return {"rollouts": [asdict(item) for item in scored], "avg_reward": _average([item.reward for item in scored])}


def self_improve(policy: LocalPatternPolicy, verifier: RuleVerifierReward, prompts: list[str], candidates: int) -> list[ReasoningExample]:
    rollouts = generate_rollouts(policy, prompts, candidates)
    scored = [Rollout(item.prompt, item.reasoning, item.answer, verifier.score(item.prompt, item.reasoning, item.answer)) for item in rollouts]
    return [ReasoningExample(item.prompt, item.reasoning, item.answer) for item in scored if item.reward > 0]


def evaluate(policy: LocalPatternPolicy, verifier: RuleVerifierReward, examples: list[ReasoningExample]) -> dict[str, float]:
    scores = []
    exact = 0
    for example in examples:
        rollout = policy.generate(example.prompt)[0]
        exact += int(rollout.answer == example.answer)
        scores.append(verifier.score(rollout.prompt, rollout.reasoning, rollout.answer))
    return {"exact_match": exact / len(examples), "avg_reward": _average(scores)}


def run_smoke_pipeline(config: dict[str, object]) -> dict[str, object]:
    start = time.time()
    run_id = f"reasoning-{uuid.uuid4().hex[:8]}"
    output_dir = Path(str(config["output_dir"])) / run_id
    train_examples = load_examples(Path(str(config["train_data"])))
    eval_examples = load_examples(Path(str(config["eval_data"])))
    policy = train_sft(train_examples, seed=int(config.get("seed", 7)))
    verifier = train_verifier(train_examples)
    prompts = [example.prompt for example in train_examples]
    grpo_report = run_grpo_step(policy, verifier, prompts, int(config.get("candidates", 3)))
    improved = self_improve(policy, verifier, prompts, int(config.get("candidates", 3)))
    policy.update(improved, learning_rate=0.1)
    eval_report = evaluate(policy, verifier, eval_examples)
    metrics = {"run_id": run_id, "backend": config.get("backend", "local"), "grpo_avg_reward": grpo_report["avg_reward"], **eval_report, "duration_seconds": round(time.time() - start, 6)}
    save_json(output_dir / "rollouts.json", grpo_report["rollouts"])
    save_json(output_dir / "self_play.json", [asdict(item) for item in improved])
    save_json(output_dir / "metrics.json", metrics)
    save_json(output_dir / "checkpoint.json", {"templates": {key: asdict(value) for key, value in policy.templates.items()}})
    save_json(output_dir / "trace.json", {"stages": ["sft", "verifier", "grpo", "self_play", "evaluation"], "config": config})
    save_json(output_dir / "dashboard.json", {"title": "reasoning-post-training-smoke", "panels": [{"metric": key, "value": value} for key, value in metrics.items() if isinstance(value, (int, float))]})
    (output_dir / "run.log").write_text("sft complete\nverifier complete\ngrpo complete\nself_play complete\nevaluation complete\n")
    return {"run_id": run_id, "output_dir": str(output_dir), "metrics": metrics}


def _average(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0
