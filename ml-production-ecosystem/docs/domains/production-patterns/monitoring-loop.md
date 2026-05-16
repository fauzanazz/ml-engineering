# Monitoring Loop Pattern

Purpose: separate endpoint metrics from production monitoring loop concepts.

Current upstream implementation:

```text
Prometheus scrapes foundation-api:8000/metrics
Grafana dashboards request/error/latency/model labels
```

Pattern boundary:

```text
metrics -> scrape -> dashboard -> review -> action
```

Later expansion can add drift dashboard, alert rules, quality gates, and runbook automation.
