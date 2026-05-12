# Step 1: Foundation Scaffold and Shared Architecture

Step pertama membuat struktur awal untuk belajar MLOps dari nol. Di tahap ini belum ada model nyata. Fokusnya adalah menyiapkan bentuk project yang bisa tumbuh dari local script menuju production pattern.

## Kenapa Step Ini Dibuat

Sebelum training model atau membuat API, project perlu punya batas arsitektur yang jelas. Tanpa batas ini, model training, deployment, logging, dan monitoring biasanya bercampur di satu script besar.

Step ini memisahkan empat komponen awal:

- model deployment
- model storage
- observability
- monitoring

## Struktur yang Dibuat

```text
ml-production-ecosystem/
├── 01-foundation/
├── 02-production-patterns/
├── 03-million-scale/
├── docs/
├── shared/
└── tests/
```

Tiga folder skala dipakai sebagai learning path. Folder `shared/` berisi kontrak yang nanti bisa dipakai ulang di semua skala.

## Shared Contracts

### Deployment

File:

```text
shared/deployment/contracts.py
```

Isi utama:

- `DeploymentResult`
- `ModelDeployer`

Tujuannya adalah membuat batas untuk komponen yang mengubah model artifact menjadi prediction endpoint atau workflow.

### Model Storage

File:

```text
shared/model_storage/contracts.py
```

Isi utama:

- `ModelArtifact`
- `ModelStore`

Tujuannya adalah menyimpan dan mengambil metadata model seperti nama, versi, URI artifact, dan lokasi metric.

### Observability

File:

```text
shared/observability/contracts.py
```

Isi utama:

- `MetricPoint`
- `ObservabilitySink`

Tujuannya adalah membuat format awal untuk metric/event yang dikirim oleh sistem ML.

### Monitoring

File:

```text
shared/monitoring/contracts.py
```

Isi utama:

- `MonitoringResult`
- `MonitoringCheck`

Tujuannya adalah membuat bentuk standar untuk hasil monitoring data, model, atau service.

## Kenapa Belum Ada Docker/Kubernetes

Foundation sengaja dimulai dari local workflow. Docker, Minikube, Kubernetes, dan Kubeflow akan lebih masuk akal setelah flow training, artifact, prediction, observability, dan monitoring lokal sudah jelas.

## Tests

Test yang ditambahkan:

```text
tests/test_shared_contracts.py
tests/test_foundation_structure.py
```

Coverage saat ini:

- shared contract bisa di-import
- docs utama tersedia
- folder skala tidak punya docs tambahan selain README
- belum ada file infra/cloud/CI/CD untuk session ini

Command:

```bash
uv run pytest
```

Result:

```text
4 passed
```

## Hasil Step 1

Step ini menghasilkan project skeleton yang siap dipakai untuk step berikutnya: membuat satu workflow ML sederhana dari training sampai prediction script.
