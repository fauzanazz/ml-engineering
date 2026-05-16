# Backpressure Pattern

Step 32 adds a small local backpressure pattern: max in-flight request limiting.

## Max In-Flight Requests

`max_in_flight` is the maximum number of requests allowed to be actively processed at the same time.

Behavior:

- if active in-flight requests are below `max_in_flight`, request is allowed
- if active in-flight requests equal `max_in_flight`, request is rejected
- slot is released after success
- slot is released after handled failure
- slot is released after exception

## Controlled Failure Response

When overloaded, return machine-readable failure instead of hanging or crashing:

```json
{
  "status": "rejected",
  "reason": "max_in_flight_reached",
  "max_in_flight": 5
}
```

Suggested HTTP status:

```text
429 Too Many Requests
```

## Why Rejecting Work Can Protect Service

A service with unlimited in-flight work can run out of CPU, memory, threads, file handles, or database connections. When too much work enters at once, all requests can become slow or fail.

Backpressure protects service by rejecting excess work early. Some callers fail fast, but allowed requests keep enough capacity to finish. Controlled failure is safer than crash, hang, or unknown timeout.

## Tradeoff

Backpressure improves stability, but rejects work. Learners should compare:

- lower `max_in_flight`: more rejection, more protection
- higher `max_in_flight`: less rejection, more overload risk

This pattern is local reliability protection. It is not autoscaling, distributed queueing, Kubernetes HPA, or real million traffic validation.

## Python Helper

`scale_reliability.backpressure.InFlightLimiter` wraps local work:

```python
from scale_reliability.backpressure import BackpressureRejected, InFlightLimiter

limiter = InFlightLimiter(max_in_flight=5)

try:
    result = limiter.run(lambda: predict(request))
except BackpressureRejected as error:
    return error.status_code, error.response
```

Use `try/finally` semantics through `limiter.run(...)` or `with limiter.slot():` so slot release happens on success, handled failure, and exception paths.
