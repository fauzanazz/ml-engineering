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
- service boundary yang lebih jelas

## Stage 3: Million Scale

Simulasikan sistem yang harus menangani request volume besar.

Target:
- load testing
- queue-based inference
- caching
- horizontal scaling
- reliability pattern
- degradation strategy

## Platform Simulation Later

Minikube, Kubernetes, dan Kubeflow dapat masuk setelah foundation dan production pattern cukup jelas.
