# ML Production Ecosystem

ML Production Ecosystem adalah project belajar MLOps dan ML Engineering dari nol. Project ini dibuat bertahap dari satu aplikasi ML sederhana, lalu naik ke pola production yang umum, sampai simulasi serving untuk request volume besar.

Fokus project bukan langsung membuat platform yang kompleks. Fokus awalnya adalah memahami komponen production ML satu per satu: model deployment, model storage, observability, monitoring, testing, dan dokumentasi keputusan teknis.

## Tujuan Project

Project ini menjawab pertanyaan utama:

- Bagaimana membawa model ML dari eksperimen lokal menjadi workflow prediction yang rapi?
- Komponen apa saja yang dibutuhkan sebelum sebuah model siap masuk production?
- Bagaimana menyimpan model artifact agar versioned, bisa ditemukan, dan bisa dipakai ulang?
- Bagaimana melakukan observability dan monitoring untuk sistem ML?
- Bagaimana arsitektur sederhana bisa berkembang menjadi medium-sized system dan high-request serving system?
- Tools MLOps apa saja yang relevan di tiap tahap, dan kapan sebaiknya dipakai?

## Learning Path

Project dibagi menjadi tiga folder berdasarkan skala:

| Scale | Folder | Fokus |
|---|---|---|
| Foundation | [`01-foundation/`](./01-foundation/) | Satu model sederhana, local/script-based deployment, shared architecture awal |
| Production Patterns | [`02-production-patterns/`](./02-production-patterns/) | Pola umum production ML seperti batch inference, online inference, registry, dan monitoring loop |
| Scale And Reliability | [`03-scale-and-reliability/`](./03-scale-and-reliability/) | Simulasi scale, reliability, load behavior, dan failure handling tanpa full cloud infra |

## Project Steps

| Step | Area | Judul | Ringkasan | Link |
|---:|---|---|---|---|
| 1 | 01 Foundation | Foundation scaffold + shared architecture | Setup struktur awal, `uv`, test skeleton, dan shared boundaries. | [docs](docs/features/step-1-foundation-scaffold-and-shared-architecture.md) |
| 2 | 01 Foundation | Foundation recommender + local infra | MovieLens recommender baseline dan local workflow awal. | [docs](docs/features/step-2-foundation-recommender-models-and-local-infra.md) |
| 3 | 01 Foundation | Local experiment tracking | Config-driven runs dan local run metadata. | [docs](docs/features/step-3-local-experiment-tracking.md) |
| 4 | 01 Foundation | Local model registry | Registry JSON lokal, model versions, active pointer. | [docs](docs/features/step-4-local-model-registry.md) |
| 5 | 01 Foundation | Local FastAPI serving | API serving active recommender model. | [docs](docs/features/step-5-local-fastapi-serving.md) |
| 6 | 01 Foundation | Serving observability basics | Local API metrics snapshot. | [docs](docs/features/step-6-serving-observability-basics.md) |
| 7 | 01 Foundation | Prediction logging + drift signal | Prediction JSONL logs dan basic drift endpoint. | [docs](docs/features/step-7-prediction-logging-and-basic-drift-signal.md) |
| 8 | 01 Foundation | Prometheus-style metrics endpoint | `/metrics` Prometheus text dan `/metrics.json` backward-compatible snapshot. | [docs](docs/features/step-8-prometheus-style-metrics-endpoint.md) |
| 9 | 01 Foundation | Local monitoring stack | Prometheus + Grafana local stack. | [docs](docs/features/step-9-local-monitoring-stack.md) |
| 10 | 01 Foundation | Dockerized Foundation API | Compose service untuk FastAPI recommender. | [docs](docs/features/step-10-dockerized-foundation-api.md) |
| 11 | 02 Transition | Batch inference job | JSONL batch recommendations; implementation masih reusable di foundation, konsep milik production patterns. | [docs](docs/features/step-11-batch-inference-job.md) |
| 12 | 02 Production Patterns | Production patterns scaffold | Pattern docs dan wrapper layer untuk online, batch, retraining, monitoring loop. | [README](02-production-patterns/README.md) |
| 13 | 02 Production Patterns | Production retraining entrypoint | `production-retrain` wrapper untuk config-driven foundation training + optional activation. | [docs](02-production-patterns/docs/retraining.md) |
| 14 | 02 Production Patterns | Model quality gate before activation | `--require-quality-gate` blocks activation unless metric thresholds pass. | [docs](02-production-patterns/docs/retraining.md) |
| 15 | 02 Production Patterns | Monitoring loop automation | `production-monitor` checks `/health`, `/metrics.json`, and `/drift`. | [docs](docs/features/step-15-monitoring-loop-entrypoint.md) |
| 16 | 02 Production Patterns | Scheduled retraining entrypoint | `production-scheduled-retrain` writes scheduler-friendly retraining reports. | [docs](docs/features/step-16-scheduled-retraining-entrypoint.md) |
| 17 | 02 Production Patterns | Airflow retraining DAG skeleton | Import-safe DAG wrapper for scheduled retraining workflow. | [docs](docs/features/step-17-airflow-dag-skeleton.md) |
| 18 | 02 Production Patterns | Alert rules + runbook | Local alert rule definitions and operational triage guide. | [docs](docs/features/step-18-alerting-rules-and-runbook.md) |
| 19 | 02 Production Patterns | Model rollback command | `production-rollback-model` restores known-good active model version. | [docs](docs/features/step-19-model-rollback-entrypoint.md) |
| 20 | 02 Production Patterns | Production model release checklist | Manual release flow for training, activation, verification, alerts, rollback. | [docs](docs/features/step-20-deployment-release-checklist.md) |
| 21 | 02 Production Patterns | Deployment manifest | Service metadata for local serving release and operations. | [docs](docs/features/step-21-deployment-manifest.md) |
| 22 | 02 Production Patterns | Lightweight CI validation | Focused local validation script for production-pattern tests. | [docs](docs/features/step-22-lightweight-ci-validation.md) |
| 23 | 02 Production Patterns | GitHub Actions CI workflow | Remote CI skeleton that runs local validation on push/PR. | [docs](docs/features/step-23-github-actions-ci-workflow.md) |
| 24 | 02 Production Patterns | Production compose profile | Production-like local `foundation-api` compose runtime. | [docs](docs/features/step-24-production-compose-profile.md) |
| 25 | 02 Production Patterns | Live API smoke test | Scripted endpoint and prediction smoke test for running `foundation-api`. | [docs](docs/features/step-25-live-api-smoke-test-script.md) |
| 26 | 02 Production Patterns | Release summary report | `production-release-summary` collects release evidence into a report. | [docs](docs/features/step-26-release-summary-report.md) |
| 27 | 02 Production Patterns | Scope review | Closure checklist for completed capabilities, known gaps, and next module boundary. | [docs](02-production-patterns/docs/scope-review.md) |
| 28 | 03 Scale And Reliability | Module scaffold and scope | Creates `03-scale-and-reliability` with scope for scale, reliability, load behavior, and failure handling. | [docs](docs/features/step-28-module-scaffold-and-scope.md) |

Supporting docs:

- [MLOps Tools Map](docs/mlops-tools-map.md)
- [Historical Run Log](docs/run-log.md)

## Current Architecture

Current flow sudah melewati foundation, menutup production patterns di Step 27, dan membuka `03-scale-and-reliability` di Step 28:

```text
MovieLens data
 -> config-driven training
 -> local artifact + experiment run
 -> local model registry + active pointer
 -> Dockerized FastAPI serving
 -> Prometheus metrics + Grafana dashboard
 -> JSONL batch inference
 -> production retraining wrapper
 -> quality gate before activation
 -> monitoring loop automation
 -> scheduled retraining + Airflow DAG skeleton
 -> alert rules + rollback + release checklist
 -> deployment manifest + local CI + GitHub Actions CI
 -> production compose + live smoke test
 -> release summary report
 -> production patterns scope review
 -> scale and reliability scaffold
```

Shared code ditempatkan di root `shared/` agar bisa dipakai ulang oleh ketiga skala project:

```text
shared/
├── deployment/
│   └── contracts.py
├── model_storage/
│   └── contracts.py
├── observability/
│   └── contracts.py
└── monitoring/
    └── contracts.py
```

## Shared Boundaries

| Boundary | Tujuan | Status |
|---|---|---|
| Deployment | Menjelaskan cara model artifact menjadi endpoint atau prediction workflow | Contract only |
| Model Storage | Menyimpan metadata model artifact seperti nama, versi, dan URI | Contract only |
| Observability | Mengirim metric/event operasional dari sistem ML | Contract only |
| Monitoring | Merepresentasikan hasil check untuk data, model, atau service health | Contract only |

## Commands

Run tests from project folder:

```bash
cd ml-production-ecosystem
uv run pytest
```

Expected result:

```text
96 passed
```

## Current Status

- `01-foundation` sudah punya train → artifact → config → experiment → registry → serving → metrics/logging → Dockerized API → monitoring stack.
- Step 11 batch inference jadi transisi ke `02-production-patterns`.
- `02-production-patterns` ditutup di Step 27 dengan scope review: retraining, quality gate, monitor, scheduled retrain, Airflow DAG skeleton, alerting, rollback, release checklist, deployment manifest, local CI, GitHub Actions CI, production compose, smoke test, release summary, dan gap checklist.
- `03-scale-and-reliability` dibuka di Step 28 sebagai scaffold scope untuk scale, reliability, load behavior, dan failure handling.
- Belum ada MLflow stages, Kubernetes, real cloud deployment, remote scheduler runtime, managed secrets, canary deployment, load testing implementation, atau real million traffic.

## Next Direction

Next step yang paling masuk akal:

1. mulai Step 29 untuk load behavior simulation di `03-scale-and-reliability`
2. queue-based inference pattern
3. caching layer untuk online recommendations
4. horizontal scaling + degradation strategy
