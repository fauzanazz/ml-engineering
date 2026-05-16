# Reliability Runbook

## Overview

This runbook is local learning guide for `scale-reliability domain`. It reuses production-patterns domain operations before adding bigger scale architecture.

Core ideas:

- monitoring detects symptoms
- rollback mitigates bad release
- retraining mitigates stale/drift model behavior
- load behavior tools identify pressure points
- not every reliability issue needs rollback
- scale fixes come after clear diagnosis

Use this runbook to triage overload, high latency, high error rate, stale model, and drift. It is not automated incident response, pager policy, Kubernetes remediation, or real production incident process.

## Signals

Primary signals:

- `production-monitor` health, metrics, drift summary
- load test report: p50/p95/max latency, error count, retry attempt count
- backpressure rejection count or `max_in_flight_reached` responses
- cache hit/miss summary
- API logs and prediction logs
- deployment manifest and release summary

Baseline monitor command:

```bash
uv run production-monitor \
  --base-url http://127.0.0.1:8000 \
  --max-error-count 0 \
  --max-drift-score 0.2 \
  --max-latency-ms-last 100
```

Smoke test command:

```bash
./scripts/smoke-test-foundation-api.sh http://127.0.0.1:8000
```

## Overload

Symptoms:

- rising latency p95
- rising error count
- rejected requests from backpressure
- retry attempt count spikes

Check:

- latest load test report
- `production-monitor` summary
- API logs
- cache hit/miss summary
- backpressure rejection count

Mitigate:

- reduce client concurrency
- disable aggressive retries
- enable or lower max in-flight limit
- use cache only when safe
- roll back if overload started after model/release change

Escalate:

- design queue, batching, or autoscaling in later steps

## High Latency

Symptoms:

- p95 or max latency exceeds SLO target
- last latency in `/metrics.json` stays high
- retries inflate end-to-end request duration

Check:

- load test report p50/p95/max
- concurrency used in test
- retry count and retry delay
- cache hit rate
- recent model or release changes from release summary

Mitigate:

- lower concurrency
- reduce retry count
- increase timeout only when service is healthy but slow
- use safe prediction cache for repeated deterministic requests
- roll back if latency regression started after a release

Scale/load mitigation helps when pressure is from too many simultaneous requests or repeated retry attempts. Rollback helps only when latency regression is tied to bad model or release change.

## High Error Rate

Symptoms:

- error count rises in load test or monitor
- HTTP 5xx responses increase
- prediction logs contain repeated failures

Check:

- `production-monitor` result
- API logs and prediction logs
- deployment manifest for command/config changes
- alerting runbook for local alert meaning
- live smoke test result

Mitigate:

- rollback bad release if errors started after deployment
- reduce load or retry pressure
- use backpressure to reject excess work before crash
- inspect input rows/requests for validation failures

4xx errors usually indicate bad input or caller behavior. 5xx errors indicate service/model/runtime failures and are more likely to need rollback or fix.

## Stale Model

Symptoms:

- active model version is old
- release summary shows no recent model promotion
- prediction quality looks stale even when service health is ok

Check:

- model registry active version
- release checklist status
- scheduled retraining output
- production-monitor drift result

Mitigate:

- run scheduled retraining if model is stale but current release is stable
- promote only after quality gate and release checklist pass
- do not rollback only because model is old unless new release caused regression

Retraining helps when stale model behavior or data drift is root cause. Rollback helps when latest model/release is bad.

## Drift

Symptoms:

- `/drift` score exceeds threshold
- `production-monitor` reports drift breach
- model output distribution changes unexpectedly

Check:

- drift score from monitor
- prediction logs used for drift sample
- recent traffic/input changes
- model version and release summary

Mitigate:

- retraining mitigates stale/drift model behavior when data changed
- rollback if drift started after bad model release
- inspect whether drift signal is model-quality risk rather than service outage

Drift is not infrastructure availability. Treat it as model-quality risk signal.

## Rollback Decision

Use rollback when evidence points to bad release or bad active model version.

Rollback command:

```bash
uv run production-rollback-model \
  --registry-path registry/models.json \
  --model-name movielens-popularity \
  --target-version foundation-config-v1 \
  --reason "reliability runbook rollback"
```

Rollback helps when:

- error rate jumped after model/release change
- latency regression started after model/release change
- drift or quality issue appears tied to latest model version
- smoke test fails after release but previous version was known-good

Rollback may not help when:

- traffic/concurrency is too high for local service
- retries amplify overload
- callers send invalid input
- feature/data distribution changed and old model is also stale

After rollback:

1. run live smoke test
2. run `production-monitor`
3. write release summary evidence
4. update release checklist with rollback reason

## Links To 02 Production Patterns

Use these production-patterns domain capabilities:

- `production-monitor` for health, metrics, and drift checks
- `production-rollback-model` for known-good model rollback
- release checklist for safe promotion/rollback decisions
- alerting runbook for local alert interpretation
- live smoke test for quick API verification
- release summary for evidence capture
- scheduled retraining for stale model or drift response

These connect immediate triage to existing learning artifacts instead of inventing a production incident process.

## Out Of Scope

This runbook excludes automated incident response, pager rotation, incident commander workflow, production escalation policy, cloud autoscaling playbook, Kubernetes remediation, database/cache outage runbooks, SLO burn-rate response, multi-region failover, and real production incident process.

## Local Autoscaling Simulation

Use `uv run scale-autoscaling-decision --load-report <load.json> --slo-report <slo.json>` after load and SLO burn-rate checks to decide whether local simulated capacity should scale up, scale down, or hold. This does not create real infrastructure.

## Distributed Load Aggregation

Use `uv run scale-aggregate-load --report <shard-1.json> --report <shard-2.json>` to combine local load-test shard reports before SLO burn-rate and autoscaling decisions. This aggregates reports only; it does not run remote workers.

## Cost Estimate

Use `uv run scale-cost-estimate --autoscaling-report <autoscaling.json> --load-report <load.json>` after autoscaling decisions to estimate local learning-unit cost. This is not cloud bill prediction.
