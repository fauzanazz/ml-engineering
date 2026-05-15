# Autoscaling Simulation

Purpose: simulate scale-up, scale-down, or hold decisions from local load and SLO reports without Kubernetes, cloud autoscaling, or managed infrastructure.

## Command

```bash
uv run scale-autoscaling-decision \
  --load-report 03-scale-and-reliability/reports/load-test.json \
  --slo-report 03-scale-and-reliability/reports/slo-burn-rate.json \
  --current-replicas 1 \
  --min-replicas 1 \
  --max-replicas 4
```

## Inputs

- load report from `scale-load-test` or `production-load-test`
- optional SLO burn-rate report from `scale-slo-burn-rate`
- current/min/max replica bounds
- target requests per replica

## Decisions

| Action | Meaning |
|---|---|
| `scale_up` | load, errors, or SLO breaches need more simulated capacity |
| `scale_down` | load and SLOs are healthy below scale-down target |
| `hold` | current replica count fits configured bounds or already at max |

## Boundary

This is local capacity-planning simulation. It does not create pods, containers, load balancer targets, cloud autoscaling policies, Kubernetes HPA objects, or production scaling guarantees.
