# MLOps Tools Map

Dokumen ini mencatat kategori tool yang akan muncul sepanjang project. Tujuannya bukan memilih semua tool di awal, tapi memahami kapan sebuah tool dibutuhkan.

## Tool Categories

| Area | Masalah yang Diselesaikan | Tool Awal | Tool Lanjutan |
|---|---|---|---|
| Environment | Membuat environment development konsisten | uv | Docker |
| Testing | Memastikan code dan struktur project tidak rusak | pytest | integration test, load test |
| Model Training | Melatih model dan menyimpan hasil eksperimen | Python script | orchestration, experiment tracking |
| Model Storage | Menyimpan model artifact dan metadata versi | local filesystem | MLflow, object storage, model registry |
| Deployment | Membuat model bisa dipakai untuk prediction | local script | FastAPI, batch job, Kubernetes service |
| Observability | Melihat behavior sistem saat berjalan | logs, simple metrics | OpenTelemetry, Prometheus, Grafana |
| Monitoring | Mendeteksi masalah data, model, dan service | custom checks | Evidently, drift monitor, alerting |
| Orchestration | Mengatur workflow multi-step | script runner | Airflow, Prefect, Kubeflow Pipelines |
| Scaling | Menangani request volume besar | local benchmark | queue, cache, autoscaling, Kubernetes |

## Urutan Belajar

1. Local foundation dulu: script, artifact, metric sederhana, test.
2. Production patterns: API, batch inference, registry, monitoring loop.
3. High-scale simulation: load test, caching, queue, horizontal scaling.
4. Platform simulation: Minikube, Kubernetes, Kubeflow, dan pattern cloud-native.

## Catatan

Tool tidak dipilih karena populer saja. Tool dipilih saat problemnya sudah jelas.
