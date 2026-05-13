# Scheduled Retraining Pattern

Purpose: run config-driven foundation training through a production-pattern entrypoint without adding Airflow yet.

Current command:

```bash
uv run production-retrain --config configs/foundation-recommender.yaml
```

Optional activation for serving:

```bash
uv run production-retrain \
  --config configs/foundation-recommender.yaml \
  --set-active
```

Explicit registry override:

```bash
uv run production-retrain \
  --config configs/foundation-recommender.yaml \
  --set-active \
  --registry-path 01-foundation/registry/models.json \
  --model-name movielens-popularity
```

Current shape:

```text
operator/cron later -> production-retrain -> train_recommender_from_config -> registry update -> optional active pointer
```

Summary stdout:

```json
{"artifact_uri":"...","metrics_uri":"...","model_name":"movielens-popularity","set_active":true,"status":"completed","version":"foundation-config-v1"}
```

Out of scope now:

- Airflow DAG
- Dockerized training job
- MLflow
- remote scheduler
- automatic promotion
