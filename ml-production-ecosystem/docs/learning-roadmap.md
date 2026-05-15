# Learning Roadmap

Roadmap ini menunjukkan urutan belajar dari local ML workflow sampai simulasi serving skala besar.

## Stage 1: Foundation

Bangun satu aplikasi ML sederhana dengan kebiasaan production sejak awal.

Target:
- training script baseline
- local model artifact
- prediction script
- metric sederhana
- monitoring check sederhana
- dokumentasi step-by-step

## Stage 2: Production Patterns

Naik dari satu workflow menjadi beberapa pola umum production ML.

Target:
- batch inference
- online inference
- model registry pattern
- feature processing pattern
- monitoring loop
- scheduled retraining
- alerting and rollback runbooks
- release checklist and deployment manifest
- production-like local runtime
- service boundary yang lebih jelas

Status: production patterns sudah punya local-first lifecycle, deployment demo, drift detection, continual-learning monitoring, rollback, local canary simulation, local scheduler dry-run, and provider boundary checks. Real cloud apply, real Kubernetes cluster apply, managed scheduler runtime, and managed secret values tetap opsional di luar local-first core.

## Stage 3: Scale And Reliability

Simulasikan sistem yang harus menangani request volume besar tanpa wajib memakai managed cloud.

Target:
- load testing
- queue-based inference
- caching
- horizontal scaling
- reliability pattern
- degradation strategy
- SLO burn-rate simulation
- autoscaling decision simulation
- distributed load aggregation
- cost estimation

## Platform Simulation Later

Minikube, Kubernetes, Kubeflow, AWS, GCP, Azure, atau vendor lain dapat masuk sebagai adapter/IaC target setelah local-first core tetap stabil.
