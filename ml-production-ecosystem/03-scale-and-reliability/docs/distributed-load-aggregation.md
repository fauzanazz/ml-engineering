# Distributed Load Aggregation

Purpose: combine multiple local `scale-load-test` shard reports into one distributed-load summary without cloud runners or a distributed test service.

## Command

```bash
uv run scale-aggregate-load \
  --report 03-scale-and-reliability/reports/load-shard-1.json \
  --report 03-scale-and-reliability/reports/load-shard-2.json \
  --output-path 03-scale-and-reliability/reports/distributed-load-aggregate.json
```

## What It Aggregates

- request, success, error, attempt, retry counts
- merged error buckets
- overall error rate
- conservative latency summary using max shard p50/p95/max and min shard min
- per-shard status summary

## Boundary

This is local aggregation for learning distributed-load behavior. It does not provision workers, coordinate remote load generators, guarantee synchronized start times, or prove real million-request capacity.
