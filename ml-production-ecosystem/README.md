# ML Production Ecosystem

ML Production Ecosystem adalah project belajar MLOps dan ML Engineering dari nol. Project ini dibuat bertahap dari satu aplikasi ML sederhana, lalu naik ke pola production yang umum, sampai simulasi serving untuk request volume besar.

Fokus project bukan langsung membuat platform yang kompleks. Fokus awalnya adalah memahami komponen production ML satu per satu: model deployment, model storage, observability, monitoring, testing, dan dokumentasi keputusan teknis.

## Tujuan Project

Project ini menjawab pertanyaan utama:

- Bagaimana membawa model ML dari eksperimen lokal menjadi workflow prediction yang rapi?
- Komponen apa saja yang dibutuhkan sebelum sebuah model siap masuk production?
- Bagaimana menyimpan model artifact agar versioned, bisa ditemukan, dan bisa dipakai ulang?
- Bagaimana melakukan observability dan monitoring untuk sistem ML?
- Bagaimana arsitektur sederhana bisa berkembang menjadi medium-sized system dan high-request serving system?
- Tools MLOps apa saja yang relevan di tiap tahap, dan kapan sebaiknya dipakai?

## Learning Path

Project dibagi menjadi tiga folder berdasarkan skala:

| Scale | Folder | Fokus |
|---|---|---|
| Foundation | [`01-foundation/`](./01-foundation/) | Satu model sederhana, local/script-based deployment, shared architecture awal |
| Production Patterns | [`02-production-patterns/`](./02-production-patterns/) | Pola umum production ML seperti batch inference, online inference, registry, dan monitoring loop |
| Million Scale | [`03-million-scale/`](./03-million-scale/) | Simulasi high-throughput serving, reliability, caching, queue, dan scaling pattern |

## Project Steps

| Step | Tanggal | Judul | Ringkasan |
|---:|---|---|---|
| 1 | 2026-05-12 | Foundation scaffold + shared architecture | Setup struktur awal, `uv`, test skeleton, dan shared boundaries untuk deployment, model storage, observability, dan monitoring. |

Detail tiap step:

- [Step 1: Foundation Scaffold and Shared Architecture](docs/features/step-1-foundation-scaffold-and-shared-architecture.md)
- [Step 9: Local Monitoring Stack With Prometheus + Grafana](docs/features/step-9-local-monitoring-stack.md)
- [Step 10: Dockerized Foundation API](docs/features/step-10-dockerized-foundation-api.md)
- [Step 11: Batch Inference Job](docs/features/step-11-batch-inference-job.md)
- [MLOps Tools Map](docs/mlops-tools-map.md)
- [Historical Run Log](docs/run-log.md)

## Current Architecture

Current flow masih berupa skeleton arsitektur, belum model training atau serving nyata:

```text
future trained model artifact
 -> model_storage contract
 -> deployment contract
 -> prediction workflow later
 -> observability metric events
 -> monitoring check results
```

Shared code ditempatkan di root `shared/` agar bisa dipakai ulang oleh ketiga skala project:

```text
shared/
├── deployment/
│   └── contracts.py
├── model_storage/
│   └── contracts.py
├── observability/
│   └── contracts.py
└── monitoring/
    └── contracts.py
```

## Shared Boundaries

| Boundary | Tujuan | Status |
|---|---|---|
| Deployment | Menjelaskan cara model artifact menjadi endpoint atau prediction workflow | Contract only |
| Model Storage | Menyimpan metadata model artifact seperti nama, versi, dan URI | Contract only |
| Observability | Mengirim metric/event operasional dari sistem ML | Contract only |
| Monitoring | Merepresentasikan hasil check untuk data, model, atau service health | Contract only |

## Commands

Run tests from project folder:

```bash
cd ml-production-ecosystem
uv run pytest
```

Expected result:

```text
4 passed
```

## Current Status

- Struktur tiga skala sudah dibuat.
- Shared architecture skeleton sudah dibuat.
- Tests untuk import contract dan layout dokumentasi sudah ada.
- Belum ada model training, prediction script, API, Docker, Kubernetes, Minikube, Kubeflow, cloud, atau CI/CD.

## Next Direction

Next step untuk foundation adalah mulai membuat satu ML workflow sederhana:

1. pilih problem dan dataset kecil
2. buat training script baseline
3. simpan model artifact secara lokal
4. buat prediction script
5. mulai catat metric dan monitoring check sederhana
