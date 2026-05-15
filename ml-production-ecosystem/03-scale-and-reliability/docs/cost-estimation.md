# Cost Estimation

Purpose: estimate local simulated serving cost from autoscaling decisions without claiming real cloud billing accuracy.

## Command

```bash
uv run scale-cost-estimate \
  --autoscaling-report 03-scale-and-reliability/reports/autoscaling-decision.json \
  --load-report 03-scale-and-reliability/reports/distributed-load-aggregate.json \
  --replica-hourly-cost 0.05 \
  --hours-per-month 730
```

## Outputs

- current and desired replicas
- replica delta
- estimated hourly cost
- estimated monthly cost
- estimated cost per 1000 requests when load report has request count

## Boundary

Currency is `learning-units`. This is local capacity/cost reasoning, not AWS/GCP/Azure pricing, cloud bill forecasting, reserved capacity modeling, or production finance reporting.
