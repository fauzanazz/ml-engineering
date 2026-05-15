# Timeout And Retry Policy

Step 31 adds client-side timeout and retry controls to `scale-load-test` for small local load simulations.

## Command

```bash
uv run scale-load-test \
  --base-url http://127.0.0.1:8000 \
  --request-count 50 \
  --concurrency 5 \
  --timeout-seconds 1 \
  --retry-count 2 \
  --retry-delay-seconds 0.1 \
  --output-path 03-scale-and-reliability/reports/load-test-retry.json
```

## What Timeout Does

`--timeout-seconds` applies to each request attempt. It prevents one attempt from waiting forever. If timeout is too aggressive, otherwise healthy slow requests can become failures and trigger retries.

## What Retry Does

`--retry-count` controls how many extra attempts an original request can make after retryable failure. `--retry-delay-seconds` waits between failed attempt and next retry.

Retry applies to:

- timeout
- connection error
- HTTP 5xx response

Retry does not apply to:

- HTTP 4xx response
- successful response

## When Retry Helps

Retry helps when failures are transient, such as one temporary timeout, one connection reset, or one short 5xx spike. Retry also helps when service still has spare capacity to handle extra attempts.

## When Retry Makes Overload Worse

Retry increases total attempt pressure. A load test with 50 original requests and `--retry-count 2` can create up to 150 attempts. If API is already overloaded, retries can make queues longer, raise p95/max latency, and increase error count.

## How To Read Report

Important fields:

- `request_count`: original requests requested by learner
- `attempt_count`: original attempts plus retry attempts
- `retry_attempt_count`: attempts beyond first try
- `retried_request_count`: original requests that needed at least one retry
- `retry_success_count`: original requests that succeeded only after retry
- `retry_exhausted_count`: original requests that still failed after retries ran out
- `errors`: grouped final failures, such as `timeout`, `connection_error`, or `http_500`

Latency metrics are end-to-end per original request and include retry attempts plus retry delay. Retry can improve `success_count` while inflating `p95` and `max` latency.

## Safe Learning Boundary

Use small retry limits and small delay/backoff. This is simulation-level client resilience, not production SLO validation, autoscaling proof, or real million traffic testing.
