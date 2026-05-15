# Local Scheduler Plan

Purpose: define local cron-compatible jobs as code before managed schedulers.

Validate without running commands:

```bash
uv run production-validate-local-scheduler
```

Boundary: this is a local scheduler contract and dry-run validator. It does not install cron entries, start Airflow, or run a distributed scheduler.
