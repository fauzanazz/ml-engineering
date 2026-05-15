# Step 40 - Reasoning Post-Training Scaffold

Added local-first, provider-agnostic DeepSeek-R1-style reasoning post-training scaffold.

Artifacts:

- `05-reasoning-post-training/reasoning_post_training/` for SFT, verifier rewards, GRPO-style updates, self-play, evaluation, backend boundaries, CLI.
- `configs/reasoning-local-smoke.yaml` for local smoke runs.
- `configs/reasoning-provider-template.yaml` for provider boundary config shape.
- `scripts/smoke-reasoning-post-training.sh` for one-command end-to-end smoke.
- `tests/test_reasoning_post_training.py` for stage, artifact, and backend-switch coverage.
- `05-reasoning-post-training/docs/reasoning-post-training.md` for algorithm, system, evaluation, failure-mode, and scaling notes.
