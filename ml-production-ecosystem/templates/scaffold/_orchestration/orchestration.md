# Orchestration Adapter

Backend: `{{backend}}`

Generated files:

- `{{adapter_filename}}` wires the scheduler/DAG shape.
- `retraining_job.py` runs the project-local training seam, quality gate, and active-model evidence.
- `monitoring_job.py` verifies the retraining report and writes monitoring evidence.

Run locally before attaching a managed scheduler:

```bash
uv run python orchestration/retraining_job.py \
  --config configs/project.yaml \
  --set-active \
  --require-quality-gate \
  --output-path artifacts/reports/scheduled-retraining.json

uv run python orchestration/monitoring_job.py \
  --input-path artifacts/reports/scheduled-retraining.json \
  --output-path artifacts/reports/monitoring.json
```

Install the real scheduler runtime only when you are ready to operate it:

- Airflow: install and copy `airflow_retraining_dag.py` into your DAGs folder.
- Metaflow: install `metaflow`, configure a datastore/compute target, then run or deploy `metaflow_retraining_flow.py`.
