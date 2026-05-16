# Batch Inference Pattern

Purpose: treat Step 11 JSONL batch inference as first production-patterns domain workflow.

Current wrapper:

```bash
uv run production-batch-recommend \
  --registry-path registry/models.json \
  --input-path examples/data/batch/input.jsonl \
  --output-path logs/batch-output.jsonl
```

Upstream implementation stays in foundation for now:

```text
src/ml_production_ecosystem/recommendation/batch.py
```

Pattern boundary:

```text
input batch file -> active registry model -> recommendations -> output batch file -> summary
```

Later expansion can add scheduler, warehouse writeback, partitioning, retries, and data contracts.
