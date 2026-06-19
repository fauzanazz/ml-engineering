# ML Production Ecosystem

Local-first toolkit untuk belajar, mencontoh, dan men-generate project ML yang siap bergerak dari eksperimen ke production workflow.

Use this command first: `ml-struct new <project-name>`.

Repo ini punya dua fungsi utama:

- **Scaffold wizard**: bikin boilerplate project ML untuk Kaggle, served model, ASR, atau enterprise pipeline.
- **Production playground**: contoh local workflow untuk training, registry, serving, validation, monitoring, rollback, reliability, dan provider boundary.

Tujuannya bukan bikin platform besar dari awal. Tujuannya bikin starting point yang rapi, bisa dites lokal, dan gampang dinaikkan levelnya sesuai kebutuhan project.

## Use Cases

| Use case | Command | Output |
|---|---|---|
| Kaggle competition | `uv run ml-struct new house-prices --preset kaggle` | training baseline, feature module, submission script, docs, tests |
| Model served as API | `uv run ml-struct new churn-api --preset served-model` | FastAPI app, prediction contract, Dockerfile, smoke-testable package |
| Generic classifier | `uv run ml-struct new churn-model --preset generic-classifier` | features, predict seam, train summary, accuracy gate |
| Batch inference | `uv run ml-struct new nightly-scorer --preset batch-inference` | batch processing seam without API assumptions |
| Existing model wrapper | `uv run ml-struct new asr-wrapper --preset existing-model-wrapper` | config + adapter around existing train/eval commands |
| Recommendation | `uv run ml-struct new recommender --preset recommendation` | candidate ranking seam and recommendation metric gate |
| LLM post-training | `uv run ml-struct new reasoner --preset llm-post-training` | dataset/evaluator seams for reasoning/LLM workflows |
| ASR served model | `uv run ml-struct new banking-asr --preset asr-served-model` | speech-to-text contract, WER/CER quality gate, FastAPI app, train/eval seam |
| Enterprise ML pipeline | `uv run ml-struct new fraud-pipeline --preset enterprise-pipeline` | ingestion-to-rollback skeleton, quality gate, runbook, tests |
| Learn production lifecycle | `uv run ml-struct quickstart` | local train → validate → approve → deploy demo → monitor evidence |

## Quickstart

```bash
cd ml-production-ecosystem
uv run ml-struct
```

Non-interactive path:

```bash
uv run ml-struct doctor
uv run ml-struct quickstart
uv run ml-struct status
uv run ml-struct explain
```

## Create New Project

Interactive wizard:

```bash
uv run ml-struct new banking-asr --preset asr-served-model
```

Preset commands:

```bash
uv run ml-struct new house-prices --preset kaggle
uv run ml-struct new churn-api --preset served-model
uv run ml-struct new churn-model --preset generic-classifier
uv run ml-struct new banking-asr --preset asr-served-model
uv run ml-struct new asr-wrapper --preset existing-model-wrapper
uv run ml-struct new recommender --preset recommendation
uv run ml-struct new nightly-scorer --preset batch-inference
uv run ml-struct new reasoner --preset llm-post-training
uv run ml-struct new fraud-pipeline --preset enterprise-pipeline
uv run ml-struct new --list-presets
```

Generated projects follow same baseline shape:

```text
project/
  README.md
  pyproject.toml
  configs/project.yaml
  data/README.md
  docs/runbook.md
  <package>/
  tests/
```


Bun commands after publishing `create-ml-struct` to npm:

```bash
# Bun create convention: resolves create-ml-struct
bun create ml-struct

# Direct package execution
bunx create-ml-struct
```


## Modular Wizard

`preset` is only starter shape. Real scaffold is composed from axes:

- `--task`: classification, regression, object-detection, segmentation, text-generation, recommendation, speech-to-text, nlp, computer-vision, forecasting, llm-post-training, batch-inference, existing-model
- `--model-type`: sklearn, xgboost, pytorch, transformers, whisper, llm, rules, external
- `--backend`: local, fastapi, batch, spark, airflow, kubernetes, serverless, external-command
- `--infra`: repeatable checklist items: api, batch, registry, quality-gate, monitoring, drift, retraining, rollback, docker, kubernetes, secrets, ci

Example: wrap existing ASR repo without moving its code:

```bash
uv run ml-struct new asr-wrapper \
  --preset existing-model-wrapper \
  --task speech-to-text \
  --model-type whisper \
  --backend external-command \
  --infra registry \
  --infra quality-gate \
  --infra monitoring
```

Generated projects include `ml-struct.yaml` and `docs/infra-checklist.md`.

## Architecture

| Area | Folder | Purpose |
|---|---|---|
| Runtime Package | `src/ml_production_ecosystem/` | importable Python code for recommendation, production patterns, reliability, reasoning, and shared contracts |
| Scaffold Templates | `templates/scaffold/` | template-first project assets with `template.yaml` metadata contracts per preset |
| Examples | `examples/samples/` | runnable sample data and command-trained example models, separate from packaged runtime code |
| Runtime State | `artifacts/`, `logs/`, `registry/` | generated local outputs kept outside source code |
| Configs | `configs/` | lifecycle, deployment, alert, platform, and provider config files |
| Docs | `docs/domains/` | moved domain READMEs and runbooks, replacing old numbered folders |
| Reasoning Data | `examples/data/reasoning-post-training/` | local reasoning SFT/RL-style sample data |

## Design Rules

- Local-first: core workflow runs without cloud credentials.
- Model-agnostic: models plug in through stable input/output contracts.
- Provider-agnostic: AWS/GCP/Azure logic belongs in adapters, not core lifecycle code.
- Evidence-driven: production readiness means generated reports, manifests, tests, and runbooks.
- Small pieces: boilerplate should teach useful boundaries without becoming framework lock-in.

## Common Commands

```bash
uv run production-lifecycle-demo --config configs/local-lifecycle-demo.yaml
uv run production-goal-readiness
uv run reasoning-post-training --config configs/reasoning-local-smoke.yaml
uv run pytest
```

Expected test result:

```text
318 passed, 3 skipped (RT warehouse integration)
```

Tip:
RT warehouse integration tests (`tests/test_rt_warehouse_foundation.py`) are skipped unless services are available. Enable and run with explicit dependency services if needed:

```bash
ML_ECOSYSTEM_RUN_RT_WAREHOUSE_TESTS=1 uv run pytest tests/test_rt_warehouse_foundation.py
```

Run full-suite with services in one command:

```bash
./scripts/validate-full-suite.sh
```

Run enterprise evidence chain in one command:

```bash
./scripts/validate-enterprise-readiness.sh
```


## Docs

- Local lifecycle: `docs/lifecycle-easy-path.md`
- Local runbook: `docs/local-lifecycle-runbook.md`
- Production patterns: `docs/domains/production-patterns/README.md`
- Reasoning post-training: `docs/domains/reasoning-post-training/reasoning-post-training.md`
- Feature history: `docs/features/`

## Current Scope

Included: local lifecycle demo, model registry, FastAPI serving, metrics/logging, Dockerized API, batch inference, retraining, quality gate, monitoring loop, scheduled retraining, rollback, release summary, SLO/load simulations, platform boundary checks, and scaffold wizard.

Not included yet: real managed cloud deployment, real Kubernetes runtime, managed secrets, distributed load execution, production Alertmanager/paging runtime, and real million-traffic autoscaling.
