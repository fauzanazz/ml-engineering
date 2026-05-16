from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from ml_production_ecosystem.reasoning_post_training.backends import ProviderBackend, backend_from_config
from ml_production_ecosystem.reasoning_post_training.core import ReasoningExample, RuleVerifierReward
from ml_production_ecosystem.reasoning_post_training.pipeline import evaluate, run_grpo_step, run_smoke_pipeline, train_sft, train_verifier
from ml_production_ecosystem.reasoning_post_training.rollout import LocalRolloutGenerator, ProviderRolloutGenerator, shard_prompts


def test_sft_verifier_grpo_and_eval_cover_major_stages() -> None:
    examples = [ReasoningExample("What is 2 plus 3?", "Compute 2 + 3. Therefore answer 5.", "5")]
    policy = train_sft(examples)
    verifier = train_verifier(examples)

    report = run_grpo_step(policy, verifier, [examples[0].prompt], candidates=2)
    metrics = evaluate(policy, verifier, examples)

    assert report["avg_reward"] > 0
    assert metrics["exact_match"] == 1.0
    assert RuleVerifierReward({examples[0].prompt: "9"}).score(examples[0].prompt, "bad", "5") < 0


def test_smoke_pipeline_writes_metrics_traces_and_checkpoint(tmp_path: Path) -> None:
    config = yaml.safe_load(Path("configs/reasoning-local-smoke.yaml").read_text())
    config["output_dir"] = str(tmp_path)

    result = run_smoke_pipeline(config)
    output_dir = Path(result["output_dir"])

    assert json.loads((output_dir / "metrics.json").read_text())["exact_match"] == 1.0
    assert (output_dir / "rollouts.json").exists()
    assert (output_dir / "self_play.json").exists()
    assert (output_dir / "checkpoint.json").exists()
    assert json.loads((output_dir / "trace.json").read_text())["stages"] == ["sft", "verifier", "grpo", "self_play", "evaluation"]
    assert json.loads((output_dir / "dashboard.json").read_text())["panels"]
    assert "grpo complete" in (output_dir / "run.log").read_text()


def test_backend_config_switches_local_and_provider_boundary() -> None:
    local_backend = backend_from_config({"backend": "local"})
    provider_backend = backend_from_config({"backend": "vendor-x", "provider_endpoint": "https://example.invalid"})

    assert local_backend.name == "local"
    assert isinstance(provider_backend, ProviderBackend)
    with pytest.raises(NotImplementedError):
        provider_backend.create_policy(seed=7)


def test_rollout_generation_has_local_shards_and_provider_boundary() -> None:
    policy = train_sft([ReasoningExample("What is 1 plus 1?", "Compute 1 + 1. Therefore answer 2.", "2")])

    assert shard_prompts(["a", "b", "c"], worker_count=2) == [["a", "c"], ["b"]]
    assert len(LocalRolloutGenerator(worker_count=2).generate(policy, ["What is 1 plus 1?"], candidates=2)) == 2
    with pytest.raises(NotImplementedError):
        ProviderRolloutGenerator("vendor-x", "rollouts").generate(policy, ["What is 1 plus 1?"], 1)
