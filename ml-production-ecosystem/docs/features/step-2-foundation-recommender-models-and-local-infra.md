---
title: Foundation Movie Recommender Models and Local Infra
type: feature-note
created: 2026-05-13
status: completed
categories: [recommendation, mlops, local-infra, artifacts]
related:
  - ./step-1-foundation-scaffold-and-shared-architecture.md
  - ../plans/2026-05-12-local-rt-warehouse-foundation-design.md
  - ../run-log.md
author: fauzan
---

# Foundation Movie Recommender Models and Local Infra

[Foundation recommender](../../01-foundation/recommendation/) sekarang punya baseline popularity, item collaborative filtering, matrix factorization, user-specific prediction, ranking evaluation, dan local Redpanda/Postgres validation.

## Context

Step sebelumnya hanya punya [MovieLens 25M](https://grouplens.org/datasets/movielens/25m/) popularity recommender. Model itu bagus sebagai baseline karena sederhana, mudah dilatih, dan menghasilkan rekomendasi global yang stabil, tapi belum menjawab kebutuhan rekomendasi personal per user.

Kita reuse pola dari [fraud-detection-pipeline](../../../fraud-detection-pipeline/) untuk memperkuat workflow ML: split data yang eksplisit, metrics object, artifact hygiene, validation mindset, dan test-first implementation. Tujuannya bukan langsung membuat recommender paling canggih, tapi membangun foundation yang bisa berkembang dari local ML script menuju production ML system.

Local production-like path juga divalidasi dengan [Redpanda](https://redpanda.com/) sebagai Kafka-compatible broker dan [PostgreSQL](https://www.postgresql.org/) sebagai warehouse. Ini memastikan training/prediction foundation tidak berdiri sendiri, tapi mulai tersambung ke pola real-time request processing dan persisted result.

## How It Works

Training tetap menulis artifact lokal ke `01-foundation/artifacts/recommendation/<version>/`. Setiap artifact berisi `model.json`, `metadata.json`, dan `metrics.json`. Prediction membaca artifact lewat `recommendation.artifacts.load_artifact()` lalu memilih path rekomendasi berdasarkan `model_name` di `model.json`.

Popularity model masih menjadi default. Model ini menghitung statistik per movie dari `ratings.csv`, lalu mengurutkan movie berdasarkan `score = positive_count * average_rating`, disusul `rating_count` dan `average_rating` sebagai tie-breaker. Output-nya global, jadi tidak membutuhkan `user_id`.

Item collaborative filtering menambahkan rekomendasi personal berbasis kemiripan item. Training membangun `user_history`, mencari movie yang disukai user dengan rating minimal `min_rating`, lalu menghitung Jaccard similarity antar movie berdasarkan overlap user yang menyukai movie tersebut. Saat prediction, sistem melihat movie yang sudah pernah dirating user, mencari movie unseen yang mirip dengan histori user, lalu mengembalikan kandidat dengan `reason = similar_to_user_history`.

Matrix factorization menambahkan model latent factor pure-Python dengan stochastic gradient descent sederhana. Training membuat faktor user dan faktor movie deterministik dengan seed tetap, lalu mengoptimalkan dot product plus `global_mean` terhadap rating historis. Saat prediction, sistem menghitung `predicted_rating` untuk movie yang belum pernah dilihat user dan mengurutkan kandidat berdasarkan skor tersebut.

Ranking evaluation ditambahkan sebagai langkah awal evaluasi recommender. `evaluate_popularity_recommender()` menghitung `precision_at_k`, `recall_at_k`, `hit_rate_at_k`, dan `coverage` dari artifact serta ratings file. Ini belum menggantikan full offline evaluation suite, tapi sudah memindahkan project dari sekadar row-count metrics menuju ranking metrics.

Local infra dijalankan lewat `docker compose up -d`. Service `redpanda` membuka Kafka-compatible endpoint di `localhost:9092`, sementara `postgres` membuka warehouse di `localhost:5432`. `foundation-rt-demo` memproduksi request event, mengonsumsi event yang sama, lalu menulis hasil proses ke table `recommendation_request_results`.

### Key Files

| File | Role |
|------|------|
| `01-foundation/recommendation/train.py` | Training untuk popularity, item collaborative filtering, dan matrix factorization. |
| `01-foundation/recommendation/predict.py` | Prediction path untuk global dan user-specific recommendations. |
| `01-foundation/recommendation/evaluate.py` | Ranking metric helpers untuk recommender artifact. |
| `01-foundation/recommendation/data.py` | MovieLens validation dan chronological train/validation/test split helper. |
| `01-foundation/recommendation/artifacts.py` | Local artifact loader/writer dan run directory helper. |
| `01-foundation/recommendation/rt_transport.py` | Kafka-compatible recommendation request event producer/consumer. |
| `01-foundation/recommendation/warehouse.py` | PostgreSQL schema and persistence for processed recommendation requests. |
| `01-foundation/recommendation/rt_demo.py` | End-to-end Redpanda + warehouse local demo. |
| `docker-compose.yml` | Local Redpanda and Postgres services. |
| `tests/test_recommendation_workflow.py` | Unit coverage for data validation, artifacts, ranking metrics, CF, and MF. |
| `tests/test_rt_warehouse_foundation.py` | Integration-style coverage for local Redpanda/Postgres path. |

### API Surface

Training functions:

```python
train_popularity_recommender(
    ratings_path: Path,
    movies_path: Path,
    artifact_dir: Path,
    version: str | None = None,
    min_rating: float = 4.0,
) -> TrainingResult
```

```python
train_collaborative_filtering_recommender(
    ratings_path: Path,
    movies_path: Path,
    artifact_dir: Path,
    version: str | None = None,
    min_rating: float = 4.0,
) -> TrainingResult
```

```python
train_matrix_factorization_recommender(
    ratings_path: Path,
    movies_path: Path,
    artifact_dir: Path,
    version: str | None = None,
    factors: int = 8,
    epochs: int = 20,
    learning_rate: float = 0.01,
    regularization: float = 0.02,
) -> TrainingResult
```

Prediction function:

```python
recommend_top_k(
    artifact_path: Path,
    top_k: int = 10,
    user_id: int | None = None,
) -> list[dict[str, object]]
```

Ranking evaluation function:

```python
evaluate_popularity_recommender(
    artifact_path: Path,
    ratings_path: Path,
    top_k: int = 10,
    min_relevant_rating: float = 4.0,
) -> RankingMetrics
```

CLI examples:

```bash
uv run foundation-train-recommender \
  --ratings-path 01-foundation/data/raw/ml-25m/ratings.csv \
  --movies-path 01-foundation/data/raw/ml-25m/movies.csv \
  --artifact-dir 01-foundation/artifacts \
  --version local-popularity-20260513 \
  --model popularity
```

```bash
uv run foundation-train-recommender \
  --ratings-path 01-foundation/data/raw/ml-25m/ratings.csv \
  --movies-path 01-foundation/data/raw/ml-25m/movies.csv \
  --artifact-dir 01-foundation/artifacts \
  --model collaborative-filtering
```

```bash
uv run foundation-train-recommender \
  --ratings-path 01-foundation/data/raw/ml-25m/ratings.csv \
  --movies-path 01-foundation/data/raw/ml-25m/movies.csv \
  --artifact-dir 01-foundation/artifacts \
  --model matrix-factorization \
  --factors 8 \
  --epochs 20
```

```bash
uv run foundation-recommend \
  --artifact-path 01-foundation/artifacts/recommendation/local-popularity-20260513 \
  --top-k 5
```

```bash
uv run foundation-recommend \
  --artifact-path 01-foundation/artifacts/recommendation/<version> \
  --user-id 10 \
  --top-k 10
```

Local infra commands:

```bash
docker compose up -d
uv run foundation-rt-demo --event-count 2 --top-k 4
uv run pytest
docker compose down
```

## Decisions & Trade-offs

We kept popularity as the default model because it is deterministic, fast on MovieLens 25M, and easy to inspect. Collaborative filtering and matrix factorization were added as separate artifact-producing trainers rather than replacing the baseline. This keeps regression risk low and makes model comparison explicit.

The collaborative filtering implementation uses simple item-to-item Jaccard similarity over positive ratings. This is readable and testable without extra dependencies, but it is not optimized for large sparse matrices. It is a foundation implementation, not a final large-scale recommender engine.

The matrix factorization implementation is pure Python SGD. This avoids adding heavy dependencies while making the model mechanics visible. The trade-off is speed: it is fine for tests and small experiments, but full MovieLens 25M training will need vectorization or a recommender library before serious scale work.

User-specific prediction requires `user_id` for collaborative filtering and matrix factorization. For unknown users, the current behavior returns an empty recommendation list instead of silently falling back to popularity. This makes cold-start behavior explicit and easier to improve later.

## Validation Results

The full MovieLens popularity model was trained locally with artifact:

```text
01-foundation/artifacts/recommendation/local-popularity-20260513
```

Top 5 recommendations from that artifact:

```text
1. Shawshank Redemption, The (1994)
2. Pulp Fiction (1994)
3. Silence of the Lambs, The (1991)
4. Forrest Gump (1994)
5. Matrix, The (1999)
```

Local infra validation ran with Redpanda and Postgres healthy. `foundation-rt-demo --event-count 2 --top-k 4` produced and consumed 2 events, then wrote 2 warehouse rows.

Test result with infra running:

```text
16 passed
```

## Known Limitations

Collaborative filtering currently stores pairwise similarities in JSON. This is not memory-efficient for large catalogs and should move to sparse storage or a nearest-neighbor index before scaling.

Matrix factorization does not yet persist user/movie bias terms, train/validation metrics, or convergence diagnostics. It also does not support incremental updates for new ratings.

Cold-start behavior is intentionally minimal. New users and new movies need a fallback strategy, likely popularity plus genre/content metadata.

Ranking evaluation currently only covers popularity artifacts. Collaborative filtering and matrix factorization need evaluator support using chronological holdout data from `load_three_way_ratings_split()`.

The real-time demo records request metadata and recommendation count, not the actual recommended movie IDs. That is enough for transport/warehouse foundation, but not enough for production recommendation auditing.

## Related

- [Step 1: Foundation Scaffold and Shared Architecture](./step-1-foundation-scaffold-and-shared-architecture.md)
- [Local RT Warehouse Foundation Design](../plans/2026-05-12-local-rt-warehouse-foundation-design.md)
- [Run Log](../run-log.md)
- [MovieLens 25M](https://grouplens.org/datasets/movielens/25m/)
- [Redpanda](https://redpanda.com/)
- [PostgreSQL](https://www.postgresql.org/)
