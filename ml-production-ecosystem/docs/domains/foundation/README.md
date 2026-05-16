# 01 Foundation

Purpose: build one simple ML application from scratch with production-oriented habits.

This stage now uses a MovieLens 25M recommendation system as the first end-to-end local ML workflow.

## Scope

Foundation is intentionally local/script-based. No API, Docker, Kubernetes, cloud, or CI/CD yet.

Current pipeline:

```text
MovieLens 25M data
 -> prepare/validate local dataset
 -> train popularity recommender
 -> save local artifact + metadata + metrics
 -> load artifact for top-k recommendations
 -> test workflow with small fixtures
```

Central docs:

- [`../README.md`](../README.md)
- [`../docs/learning-roadmap.md`](../docs/learning-roadmap.md)
- [`../docs/mlops-tools-map.md`](../docs/mlops-tools-map.md)
- [`../docs/run-log.md`](../docs/run-log.md)
- [`../docs/features/step-1-foundation-scaffold-and-shared-architecture.md`](../docs/features/step-1-foundation-scaffold-and-shared-architecture.md)
- [`../docs/features/step-9-local-monitoring-stack.md`](../docs/features/step-9-local-monitoring-stack.md)
- [`../docs/features/step-10-dockerized-foundation-api.md`](../docs/features/step-10-dockerized-foundation-api.md)
- [`../docs/features/step-11-batch-inference-job.md`](../docs/features/step-11-batch-inference-job.md)

## Local Paths

Ignored local runtime paths:

```text
examples/data/foundation/
artifacts/foundation/
```

These paths hold downloaded MovieLens files and generated model artifacts. They should not be committed.

## Runtime Commands

Run commands from project root:

```bash
cd ml-production-ecosystem
```

Prepare MovieLens 25M data:

```bash
uv run foundation-prepare-data --data-dir examples/data/foundation
```

Train baseline recommender:

```bash
uv run foundation-train-recommender \
  --ratings-path examples/samples/recommendation/ratings.csv \
  --movies-path examples/samples/recommendation/movies.csv \
  --artifact-dir artifacts/foundation \
  --version foundation-flow-test
```

Generate top-k recommendations:

```bash
uv run foundation-recommend \
  --artifact-path artifacts/foundation/recommendation/foundation-flow-test \
  --top-k 10
```

Run local production-like RT transport + warehouse demo:

```bash
docker compose up -d
uv run foundation-rt-demo
docker compose down
```

Run tests. These tests expect Docker services from `docker compose up -d` to be available:

```bash
uv run pytest
```

## Current Model

Model type: popularity baseline recommender.

Inputs:

- `ratings.csv`
- `movies.csv`

Output artifact:

```text
artifacts/foundation/recommendation/<version>/
├── metadata.json
├── metrics.json
└── model.json
```

The recommender ranks movies using rating counts, positive-rating counts, and average rating. It is simple by design: useful as a baseline and easy to monitor before adding more advanced recommendation methods.

## Example Result

A full MovieLens 25M run returned top recommendations such as:

1. `Shawshank Redemption, The (1994)`
2. `Pulp Fiction (1994)`
3. `Silence of the Lambs, The (1991)`
4. `Forrest Gump (1994)`
5. `Matrix, The (1999)`

## Local Production-Like RT Foundation

The foundation stage includes a local production-like streaming path:

```text
recommendation request producer
 -> Redpanda Kafka-compatible topic
 -> Python consumer
 -> PostgreSQL warehouse table
 -> SQL readback
```

Concept mapping:

| Local component | Production concept |
|---|---|
| Redpanda | Kafka-compatible real-time transport |
| PostgreSQL | Warehouse stand-in for queryable durable records |
| `foundation.recommendation.requests` | Kafka topic/stream |
| `foundation-rt-demo` | End-to-end producer/consumer/warehouse workflow |

## Target Learning Outcomes

- Learn MLOps from scratch using one simple model project.
- Understand data preparation, model training, artifact storage, prediction, simple metrics, and validation.
- Build reusable foundations for later streaming simulation, drift detection, production patterns, and million-scale serving.
- Document ML Engineering best practices in a portfolio-friendly way.

## Status

Foundation now has a simple working local recommendation pipeline:

- MovieLens 25M download/unpack/validation command works.
- Popularity recommender training command works.
- Local artifact generation works.
- Recommendation command works.
- Tests use tiny fixtures and do not require network or full MovieLens data.

Current verification requires local Docker services for RT/warehouse tests:

```bash
docker compose up -d
uv run pytest
uv run foundation-rt-demo
docker compose down
```
