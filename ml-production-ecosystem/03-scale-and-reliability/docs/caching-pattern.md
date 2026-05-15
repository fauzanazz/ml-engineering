# Caching Pattern

Step 34 adds a small in-memory prediction cache pattern for repeated inference requests.

Caching can reduce latency and load when identical requests repeat. It can also be dangerous for ML systems when the cached answer no longer matches model, feature, user, permission, or policy state.

## What The Helper Does

`scale_reliability.cache.PredictionCache`:

- creates deterministic cache keys from normalized request payloads
- records cache hits
- records cache misses
- does not cache failed predictions
- reports hit/miss summary
- keeps storage bounded with `max_entries`

Example summary:

```json
{
  "cache": {
    "hit_count": 12,
    "miss_count": 8,
    "entry_count": 8,
    "max_entries": 100,
    "hit_rate": 0.6
  }
}
```

## Safe Cases

Prediction caching is safer when:

- model output is deterministic
- model version is stable or included in cache key
- response is read-only recommendation data
- requests are repeated anonymous/public requests
- TTL is short or cache invalidates on model-version change
- payload contains no volatile context
- cache key includes all fields that affect prediction output

## Dangerous Cases

Prediction caching is dangerous when:

- output is personalized and user context changes
- model version changes but cache key does not include model version
- feature data becomes stale
- regulated decisions require fresh evaluation
- permissions or entitlements can change
- model output is non-deterministic
- cache key misses important input fields
- sensitive data is stored in cache

## Why ML Cache Needs Extra Care

Web caching often treats identical input as identical output. ML serving may depend on hidden context: model version, feature freshness, user state, policy, experiment group, permissions, or time. If those values affect prediction, they must be in the cache key or the response must not be cached.

## Local Boundary

This helper is local in-memory cache only. It is not Redis, distributed invalidation, TTL scheduler, CDN caching, privacy policy enforcement, feature store freshness integration, or production cache tuning.
