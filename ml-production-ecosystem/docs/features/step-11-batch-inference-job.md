# Step 11: Batch Inference Job

## Goal

Add offline JSONL batch prediction path for the foundation recommender, alongside online FastAPI serving.

## Command

```bash
uv run foundation-batch-recommend \
  --registry-path 01-foundation/registry/models.json \
  --input-path 01-foundation/data/batch/input.jsonl \
  --output-path 01-foundation/logs/batch-output.jsonl
```

## Input JSONL

Each line is one request:

```json
{"request_id": "batch-1", "user_id": 10, "top_k": 5}
```

Fields:

| Field | Required | Notes |
|---|---:|---|
| `request_id` | No | Defaults to row id if missing. |
| `user_id` | No | Required only by personalized model types. |
| `top_k` | No | Defaults to `10`; must be positive. |

## Output JSONL

Success row:

```json
{"request_id":"batch-1","model_name":"movielens-popularity","version":"api-v1","recommendations":[],"error":null}
```

Error row:

```json
{"request_id":"bad-1","model_name":"movielens-popularity","version":"api-v1","recommendations":[],"error":"top_k must be positive, got 0"}
```

Per-row errors do not fail the whole job. Invalid config or missing input file still fails the job.

## Summary stdout

Command prints JSON summary:

```json
{"failed":0,"input_rows":1,"output_path":"01-foundation/logs/batch-output.jsonl","succeeded":1}
```

## Pattern

```text
JSONL request file
  -> active model from local registry
  -> recommend_top_k_from_registry
  -> JSONL output file
  -> stdout summary
```

## Scope Notes

- JSONL only for now.
- No Airflow, Spark/Ray, parallel workers, warehouse writeback, or scheduling yet.
- Dockerized batch job out of scope.
