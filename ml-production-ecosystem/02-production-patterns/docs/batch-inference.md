# Batch Inference Pattern

Purpose: treat Step 11 JSONL batch inference as first 02-production-patterns workflow.

Current wrapper:

```bash
uv run production-batch-recommend \
  --registry-path 01-foundation/registry/models.json \
  --input-path 01-foundation/data/batch/input.jsonl \
  --output-path 01-foundation/logs/batch-output.jsonl
```

Upstream implementation stays in foundation for now:

```text
01-foundation/recommendation/batch.py
```

Pattern boundary:

```text
input batch file -> active registry model -> recommendations -> output batch file -> summary
```

Later expansion can add scheduler, warehouse writeback, partitioning, retries, and data contracts.
