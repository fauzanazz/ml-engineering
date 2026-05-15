# Batch Performance Baseline

Step 11 proved batch inference exists. It reads JSONL input, writes JSONL prediction output, and keeps row-level failures from stopping whole batch run.

Step 33 measures batch behavior. Batch inference can now write a performance report with input size, duration, success/error rows, and throughput rows/sec.

## Command

```bash
uv run production-batch-recommend \
  --registry-path 01-foundation/registry/models.json \
  --input-path 01-foundation/data/batch/input.jsonl \
  --output-path 01-foundation/logs/batch-output.jsonl \
  --report-path 03-scale-and-reliability/reports/batch-performance.json
```

## Report

```json
{
  "input_path": "01-foundation/data/batch/input.jsonl",
  "output_path": "01-foundation/logs/batch-output.jsonl",
  "input_row_count": 1000,
  "success_row_count": 1000,
  "error_row_count": 0,
  "duration_seconds": 2.41,
  "throughput_rows_per_second": 414.94,
  "status": "passed"
}
```

## Why Input Size Matters

Throughput rows/sec depends on input size. A 10-row local batch can be dominated by startup and file I/O overhead. Larger local inputs give a more useful baseline for comparing later changes, but still do not prove production-scale behavior.

## Why Throughput Helps

Rows/sec gives one simple comparison metric across runs:

- same code, bigger input
- same input, changed model logic
- same input, later optimization
- same input, different machine

Use it with duration and error rows. High throughput with many failed rows is not a good batch result.

## Local Baseline Boundary

This is local baseline measurement, not production-scale batch benchmarking. It does not include Spark, Ray, Dask, distributed workers, feature store reads, cloud object storage, scheduler tuning, autoscaling, or million-row production claims.
