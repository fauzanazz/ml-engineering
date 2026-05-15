# SLO Definition

Step 35 defines simple local reliability targets for `03-scale-and-reliability`. These are learning defaults, not production commitments.

## SLO, SLI, Alert

- SLO is target behavior, not a guarantee.
- SLI is measurement signal used to evaluate service behavior.
- alert is operational notification when a signal crosses a threshold.

Local project SLOs are learning defaults. These thresholds should be tuned with real traffic later. drift threshold is model-quality risk signal, not infrastructure availability. There is no burn-rate math yet.

## SLO Table

| SLO | Target | Source | Breach Signal |
|---|---:|---|---|
| Availability | >= 99% local check pass rate | `production-monitor` health check | `/health` check fails |
| Latency p95 | <= 200 ms | load test report / `/metrics.json` | p95 latency exceeds threshold |
| Error Rate | <= 1% | load test report / monitor error count | error rate exceeds threshold |
| Drift Threshold | <= 0.2 drift score | `/drift` and `production-monitor` | drift score exceeds threshold |

## Alert To SLO Breach Mapping

- service health alert maps to availability SLO breach.
- high latency alert maps to latency p95 SLO breach.
- error count or error rate alert maps to error rate SLO breach.
- drift alert maps to drift threshold SLO breach.

## Example Mapping

SLO: Latency p95 <= 200 ms

SLI: p95 latency from load test report

Alert: high latency alert when p95 > 200 ms

Action: inspect load behavior, concurrency, cache hit rate, and downstream latency.

## Notes Per SLO

### Availability

Availability measures whether serving API health checks pass. In this local project, it is based on repeated `production-monitor` health checks, not external synthetic monitoring.

### Latency p95

Latency p95 measures tail latency. It is more useful than average latency when comparing sequential load, concurrent load, retry behavior, cache hit rate, and backpressure.

### Error Rate

Error Rate measures failed requests over total requests. Load test reports and monitor error counts can both indicate breach risk.

### Drift Threshold

Drift Threshold measures model-quality risk. Drift can mean input or prediction behavior changed. It does not mean the server is unavailable.

## Out Of Scope

This step intentionally excludes SLO burn-rate math, error budget calculation, multi-window alerting, paging policy, production incident process, Prometheus alert rule rewrite, dashboards, autoscaling policy, Kubernetes production config, and real production SLO commitment.
