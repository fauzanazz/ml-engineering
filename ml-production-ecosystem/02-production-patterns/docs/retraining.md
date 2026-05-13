# Scheduled Retraining Pattern

Purpose: reserve seam for scheduled retraining without adding Airflow yet.

Near-term shape:

```text
scheduler/cron later -> retrain entrypoint -> config training -> registry update -> optional promotion gate
```

Do not implement scheduler here yet. Next step can add local retraining skeleton or quality gate once production-pattern scaffold is stable.

Out of scope now:

- Airflow DAG
- Dockerized training job
- MLflow
- remote scheduler
- automatic promotion
