---
title: AWS Serving Decision
type: architecture-decision
created: 2026-05-15
status: completed
categories: [ml-production, aws, serving, architecture]
related:
  - ./deployment-manifest.md
  - ./release-checklist.md
  - ./production-compose.md
---

# AWS Serving Decision

This document explains when this project would use ECS, EKS, or SageMaker for model serving. It is architecture reasoning only; current implementation remains local Docker Compose and does not deploy to AWS.

## Current System Shape

`foundation-api` is a containerized FastAPI model serving service with:

- `/health` for service health.
- `/metrics` for Prometheus-style metrics.
- `/metrics.json` for machine-readable monitor checks.
- `/drift` for basic drift signal.
- local model registry at `registry/models.json`.
- local artifact storage under `artifacts/foundation`.

Local production-like runtime is documented in [production-compose.md](./production-compose.md). Release and rollback workflow is documented in [release-checklist.md](./release-checklist.md).

## Decision Summary

For this project stage, the preferred AWS target would be **ECS on Fargate**.

Reason: serving shape is a single containerized API with simple health checks, logs, metrics, and horizontal scaling needs. ECS gives enough operational realism without adding Kubernetes platform ownership too early.

Use **EKS** later when Kubernetes-specific platform skills become the learning goal or when multiple services need shared cluster primitives.

Use **SageMaker** when managed ML serving, model registry integration, deployment variants, or managed inference operations matter more than owning the service runtime.

## Option 1: ECS on Fargate

Best fit when:

- Workload is one or few containerized APIs.
- Team wants low cluster-operations burden.
- Service needs autoscaling, load balancing, IAM task roles, CloudWatch logs, and deployment health checks.
- Kubernetes APIs are not required.

How this project maps:

| Project concept | ECS/Fargate mapping |
|---|---|
| Docker image | ECR image |
| `foundation-api` | ECS service |
| `/health` | ALB target group health check |
| `/metrics` | CloudWatch agent, ADOT, or Prometheus-compatible scraping path |
| model artifacts | S3 object paths mounted/read at startup or pulled during build |
| registry metadata | DynamoDB, S3 JSON, or MLflow-backed metadata |
| rollback | ECS task definition rollback or model registry active pointer rollback |
| secrets | AWS Secrets Manager or SSM Parameter Store |

Pros:

- Small operational surface.
- Good for first production container deployment.
- IAM task roles support least privilege cleanly.
- Fits FastAPI service without Kubernetes YAML.

Cons:

- Less portable than Kubernetes manifests.
- Advanced traffic shaping and service mesh patterns are less natural than EKS.
- ML-specific workflows still need separate orchestration.

Decision: **best near-term AWS target** for current service shape.

## Option 2: EKS

Best fit when:

- Kubernetes itself is target platform skill.
- System has multiple services, shared ingress, HPA, ConfigMaps, Secrets, and common observability stack.
- Team needs portability across Kubernetes environments.
- Platform team already operates clusters.

How this project maps:

| Project concept | EKS mapping |
|---|---|
| Docker image | ECR image used by Deployment |
| `foundation-api` | Kubernetes Deployment + Service |
| `/health` | readiness/liveness probes |
| `/metrics` | Prometheus scrape target |
| scaling | HPA based on CPU/custom metrics |
| config | ConfigMap |
| secrets | Kubernetes Secret backed by AWS Secrets Manager via CSI driver |
| rollback | Deployment rollout undo or active model pointer rollback |

Pros:

- Strong portfolio evidence for production platform operations.
- Natural fit for HPA, probes, ConfigMap, Secret, and Prometheus.
- Useful for MLOps interviews where platform reasoning matters.

Cons:

- Higher operational complexity.
- Requires cluster security, node/pod IAM, ingress, and observability decisions.
- Easy to overbuild for one service.

Decision: **good next learning phase**, not first AWS deployment target unless Kubernetes proof is explicit goal.

## Option 3: SageMaker Real-Time Endpoint

Best fit when:

- Managed ML inference is preferred over general web-service ownership.
- Model registry, endpoint variants, canary/shadow deployment, and managed autoscaling are valuable.
- Team wants AWS-native ML operations and less custom serving code.
- Framework/container fits SageMaker inference contract.

How this project maps:

| Project concept | SageMaker mapping |
|---|---|
| trained model artifact | S3 model artifact |
| model version | SageMaker model package or model registry entry |
| FastAPI serving code | custom inference container or SageMaker inference script |
| active model | endpoint config / production variant |
| rollback | switch endpoint config or production variant |
| metrics/logs | CloudWatch metrics/logs |

Pros:

- Managed ML deployment lifecycle.
- Native model registry and endpoint deployment patterns.
- Less infrastructure ownership.

Cons:

- Requires adapting serving interface.
- Less representative of generic microservice deployment.
- Can hide some operational mechanics useful for learning.

Decision: **strong for ML-native managed serving**, but not ideal for this project stage because current learning goal is production system mechanics.

## Recommendation By Scenario

| Scenario | Choose | Reason |
|---|---|---|
| First AWS deployment for current FastAPI service | ECS Fargate | Smallest production-realistic container path |
| Need Kubernetes hands-on proof | EKS or local Minikube first | Proves probes, HPA, ConfigMap, Secret, Service |
| Enterprise already standardized on Kubernetes | EKS | Fits platform conventions |
| Need managed ML endpoint with variants | SageMaker | Native ML serving lifecycle |
| Need custom API plus business logic | ECS or EKS | More control over service behavior |
| Need lowest platform ownership | SageMaker or ECS Fargate | Less cluster management |

## Interview Framing

A good answer is not “EKS is better” or “SageMaker is better.” Better answer:

```text
I would choose ECS Fargate first for this project because the workload is currently a single containerized FastAPI service with health checks, metrics, artifact loading, and rollback through registry state. EKS becomes justified when Kubernetes primitives are required or when platform consistency matters. SageMaker becomes justified when managed ML deployment lifecycle, model registry integration, endpoint variants, and AWS-native inference operations matter more than custom service ownership.
```

## Future Work

- Add `k8s/` manifests for local Kubernetes proof.
- Add ECR image push and ECS task definition sketch.
- Add SageMaker inference adapter design note if managed endpoint path becomes target.
- Add cost and scaling comparison after load testing exists in `scale domain`.
