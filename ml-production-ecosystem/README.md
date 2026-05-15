# ML Production Ecosystem

Local-first toolkit untuk belajar, mencontoh, dan men-generate project ML yang siap bergerak dari eksperimen ke production workflow.

Repo ini punya dua fungsi utama:

- **Scaffold wizard**: bikin boilerplate project ML untuk Kaggle, served model, atau enterprise pipeline.
- **Production playground**: contoh local workflow untuk training, registry, serving, validation, monitoring, rollback, reliability, dan provider boundary.

Tujuannya bukan bikin platform besar dari awal. Tujuannya bikin starting point yang rapi, bisa dites lokal, dan gampang dinaikkan levelnya sesuai kebutuhan project.

## Use Cases

| Use case | Command | Output |
|---|---|---|
| Kaggle competition | `uv run mle new --preset kaggle` | training baseline, feature module, submission script, docs, tests |
| Model served as API | `uv run mle new --preset served-model` | FastAPI app, prediction contract, Dockerfile, smoke-testable package |
| Enterprise ML pipeline | `uv run mle new --preset enterprise-pipeline` | ingestion-to-rollback skeleton, quality gate, runbook, tests |
| Learn production lifecycle | `uv run mle quickstart` | local train → validate → approve → deploy demo → monitor evidence |

## Quickstart

```bash
cd ml-production-ecosystem
uv run mle
```

Non-interactive path:

```bash
uv run mle doctor
uv run mle quickstart
uv run mle status
uv run mle explain
```

## Create New Project

Interactive wizard:

```bash
uv run mle new
```

Preset commands:

```bash
uv run mle new --preset kaggle --name house-prices --target ../house-prices
uv run mle new --preset served-model --name churn-api --target ../churn-api
uv run mle new --preset enterprise-pipeline --name fraud-pipeline --target ../fraud-pipeline
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

## Architecture

| Area | Folder | Purpose |
|---|---|---|
| Foundation | `01-foundation/` | local recommender workflow, model artifacts, registry, serving, metrics |
| Production Patterns | `02-production-patterns/` | lifecycle wrappers, scaffold wizard, batch/retrain/monitor/rollback patterns |
| Scale And Reliability | `03-scale-and-reliability/` | load, retry, backpressure, caching, SLO, cost, readiness simulations |
| Platform And Cloud | `04-platform-and-cloud/` | provider-neutral boundaries, adapter/IaC scope, secret references |
| Reasoning Post-Training | `05-reasoning-post-training/` | local reasoning SFT/RL-style smoke workflow and provider adapter boundary |
| Shared Contracts | `shared/` | model, lifecycle, storage, observability, monitoring, deployment, platform ports |

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
278 passed
```

## Docs

- Local lifecycle: `docs/lifecycle-easy-path.md`
- Local runbook: `docs/local-lifecycle-runbook.md`
- Production patterns: `02-production-patterns/README.md`
- Reasoning post-training: `05-reasoning-post-training/docs/reasoning-post-training.md`
- Feature history: `docs/features/`

## Current Scope

Included: local lifecycle demo, model registry, FastAPI serving, metrics/logging, Dockerized API, batch inference, retraining, quality gate, monitoring loop, scheduled retraining, rollback, release summary, SLO/load simulations, platform boundary checks, and scaffold wizard.

Not included yet: real managed cloud deployment, real Kubernetes runtime, managed secrets, distributed load execution, production Alertmanager/paging runtime, and real million-traffic autoscaling.
