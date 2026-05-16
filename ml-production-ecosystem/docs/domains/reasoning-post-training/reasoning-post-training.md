# DeepSeek-R1-Style Reasoning Post-Training

This module is a local-first learning scaffold, not a DeepSeek-R1 reproduction. It shows production-shaped boundaries for reasoning post-training with deterministic toy data.

## Pipeline

1. SFT loads reasoning examples and fits a local pattern policy.
2. Verifier reward model maps prompts to reference answers and scores reasoning/answer pairs.
3. GRPO-style step samples grouped rollouts per prompt, scores them, and updates from best positive samples.
4. Self-improvement keeps high-reward generated traces as new SFT data.
5. Evaluation reports exact match and average verifier reward.

## Systems Design

- Local backend runs without network, GPUs, or provider SDKs.
- Provider backend and rollout generator are adapter boundaries only; configs can switch backend names without importing vendor SDKs.
- Runs emit `metrics.json`, `trace.json`, `dashboard.json`, `run.log`, `rollouts.json`, `self_play.json`, and `checkpoint.json` under `artifacts/reasoning-runs/<run_id>/`.
- Dataset paths, seed, candidate count, backend, and output path live in YAML config for reproducibility.

## Smoke Command

```bash
uv run reasoning-post-training --config configs/reasoning-local-smoke.yaml
```

or:

```bash
./scripts/smoke-reasoning-post-training.sh
```

## Failure Modes

- Verifier overfits references and rewards shortcut answers.
- Local pattern policy cannot generalize beyond simple arithmetic templates.
- Self-play can amplify bad verifier judgments.
- Provider adapters need queueing, retries, auth, rate limits, and cost controls before real remote rollout generation.

## Scaling Bottlenecks

- Rollout generation dominates cost and latency as candidates per prompt grow.
- Verifier inference becomes hot path during GRPO and self-play filtering.
- Checkpoint and rollout logs grow quickly; production runs need object storage, retention, and sharding.
- Distributed rollouts need idempotent work units and deterministic run metadata.
