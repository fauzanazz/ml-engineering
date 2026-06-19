# Local-First Lifecycle Easy Path

Purpose: make common ML production actions easy without tying core workflow to one model type or cloud provider.

This is the target operator experience:

```text
add data
  -> train model automatically
  -> validate model offline
  -> approve deployment
  -> run deployment demo test
  -> detect drift
  -> monitor continual-learning decision
  -> inspect graph-based lifecycle flow
```

## Contract First

Every step should be replaceable behind shared contracts:

| Step | Shared contract | Current local proof |
|---|---|---|
| Validate model contract | `ModelIOContract`, `PredictionPort` | model contract manifest and JSON schemas |
| Validate platform plan | `InfrastructurePlan`, `SecretRef` | local IaC plan validation with secret references only |
| Add data | `DatasetRef`, `DataIngestionPort` | dataset manifest and MovieLens validation |
| Train model automated | `RetrainingPort`, `LifecycleRun` | `uv run production-retrain --config configs/foundation-recommender.yaml` |
| Validate offline | `OfflineValidationReport`, `OfflineValidationPort` | offline validation report and quality gate |
| Approve deployment | `ApprovalPort`, `ReleaseDecision` | approval decision artifact and active model pointer |
| Run deployment demo test | `DeploymentDemoResult`, `DeploymentDemoPort` | deployment demo report, production compose, and smoke test |
| Get drift detection | `DriftReport`, `DriftDetectionPort` | drift report, `/drift` endpoint, and `production-monitor` |
| Monitor continual learning | `ContinualLearningDecision`, `ContinualLearningPort` | continual-learning decision, scheduled retraining, and monitoring loop |
| Graph interactive flow | `LifecycleGraph`, `LifecycleGraphPort` | Mermaid and local HTML artifact graph |

Current contracts live in `src/ml_production_ecosystem/shared/lifecycle/contracts.py`.

Serving adapters use `PredictionPort`, `PredictionRequest`, and `PredictionResponse` from `src/ml_production_ecosystem/shared/model_contracts/contracts.py`. The local recommender has a bridge adapter at `src/ml_production_ecosystem/recommendation/prediction_adapter.py`, so legacy model code can expose generic in/out contracts without changing API clients.

## Local Commands Today

Validate model contract:

```bash
uv run production-validate-model-contract --config configs/foundation-recommender.yaml
```

Validate local platform plan:

```bash
uv run production-validate-platform-plan
```

Validate dataset:

```bash
uv run production-ingest-data --config configs/foundation-recommender.yaml
```

Then train and optionally activate after quality gate:

```bash
uv run production-retrain \
  --config configs/foundation-recommender.yaml \
  --set-active \
  --require-quality-gate
```

For non-recommender models, `production-retrain` also accepts command-based training configs:

```yaml
training:
  type: command
  command: ["python", "train_any_model.py", "--summary-path", "reports/training-summary.json"]
  summary_path: reports/training-summary.json
```

The command must write JSON with `model_name`, `version`, `artifact_uri`, and `metrics_uri`. This keeps core lifecycle logic generic while the model-specific trainer lives outside core code.

Framework-specific seams reuse the same command boundary (`command`, `summary_path`) while making intent explicit in config:

```yaml
training:
  type: onnx
  framework: onnx
  command:
    - python
    - examples/onnx/train.py
    - --summary-path
    - artifacts/reports/production-patterns/onnx-training-summary.json
  summary_path: artifacts/reports/production-patterns/onnx-training-summary.json

training:
  type: pytorch
  framework: pytorch
  command:
    - python
    - examples/pytorch/train.py
    - --summary-path
    - artifacts/reports/production-patterns/pytorch-training-summary.json
  summary_path: artifacts/reports/production-patterns/pytorch-training-summary.json
```

These are first-class minimum hooks: as long as your training command writes the required summary contract, no edit to core lifecycle or retraining code is required.

Runnable non-recommender proof:

```bash
uv run production-validate-model-contract --config configs/generic-classifier-command.yaml
uv run production-retrain \
  --config configs/generic-classifier-command.yaml \
  --set-active \
  --require-quality-gate
```

This trains `examples/samples/generic_classifier/train.py` through command adapter, validates classification schemas in `schemas/generic_classifier/`, writes local artifacts, and activates `tiny-threshold-classifier` without recommender-specific training code.

Write standalone offline validation report:

```bash
uv run production-validate-offline --config configs/foundation-recommender.yaml
```

Create explicit approval decision:

```bash
uv run production-approve-model \
  --validation-report artifacts/reports/production-patterns/offline-validation.json \
  --approve
```

Run local production-like API:

```bash
docker compose -f docker-compose.production.yaml up --build foundation-api
```

Smoke test deployment:

```bash
./scripts/smoke-test-foundation-api.sh http://127.0.0.1:8000
```

Write machine-readable deployment demo report:

```bash
uv run production-demo-deployment --base-url http://127.0.0.1:8000
```

Write standalone drift report:

```bash
uv run production-detect-drift --base-url http://127.0.0.1:8000
```

Write continual-learning decision:

```bash
uv run production-continual-decision \
  --drift-report artifacts/reports/production-patterns/drift-report.json \
  --deployment-demo artifacts/reports/production-patterns/deployment-demo.json \
  --history-path artifacts/reports/production-patterns/continual-learning-history.jsonl
```

`continual-learning-history.jsonl` appends decisions over repeated checks. Summarize it locally with `uv run production-continual-summary --history-path artifacts/reports/production-patterns/continual-learning-history.jsonl`, giving local continual-learning monitoring trends without managed services.

Write lifecycle graph artifacts:

```bash
uv run production-lifecycle-graph
```

Summarize local lifecycle status:

```bash
uv run production-lifecycle-status
```

Monitor health, latency, errors, and drift:

```bash
uv run production-monitor \
  --base-url http://127.0.0.1:8000 \
  --max-error-count 0 \
  --max-drift-score 0.2 \
  --max-latency-ms-last 100
```

## One-Command Flow

Current ergonomic command:

```bash
uv run production-lifecycle-demo --config configs/foundation-recommender.yaml
```

No-download local demo command:

```bash
uv run production-lifecycle-demo --config configs/local-lifecycle-demo.yaml
```

No-download local deployment smoke:

```bash
./scripts/smoke-local-deployment.sh
```

This starts the local API, sends a prediction, validates deployment health, and writes a drift report without cloud services.

Default behavior:

```text
validate dataset
  -> validate model contract
  -> validate platform plan
  -> write dataset manifest
  -> train candidate
  -> write offline validation report
  -> create pending approval decision
  -> skip activation until approved
  -> skip deployment demo until base URL is provided
  -> write lifecycle graph
```

Approve and activate after quality gate:

```bash
uv run production-lifecycle-demo \
  --config configs/foundation-recommender.yaml \
  --approve \
  --set-active
```

Include running local API checks:

```bash
uv run production-lifecycle-demo \
  --config configs/foundation-recommender.yaml \
  --approve \
  --set-active \
  --base-url http://127.0.0.1:8000
```

Outputs:

```text
artifacts/reports/production-patterns/lifecycle-demo.json
artifacts/reports/production-patterns/lifecycle-demo.mmd
artifacts/reports/production-patterns/lifecycle-demo.html
artifacts/reports/production-patterns/platform-plan-validation.json
artifacts/reports/production-patterns/model-contract-manifest.json
artifacts/reports/production-patterns/dataset-manifest.json
artifacts/reports/production-patterns/offline-validation.json
artifacts/reports/production-patterns/approval-decision.json
artifacts/reports/production-patterns/deployment-demo.json
artifacts/reports/production-patterns/drift-report.json
artifacts/reports/production-patterns/continual-learning-decision.json
artifacts/reports/production-patterns/continual-learning-history.jsonl
artifacts/reports/production-patterns/continual-learning-summary.json
```

The command stays local-first. Provider adapters may add AWS/GCP/Azure deployment targets later, but must not change model contracts or core lifecycle semantics.

## Graph Target

Graph output includes Mermaid for docs and a local HTML artifact with links to lifecycle reports:

```mermaid
flowchart LR
    data["Add Data"]
    train["Train Candidate"]
    validate["Offline Validation"]
    approve["Approve Deployment"]
    demo["Deployment Demo Test"]
    drift["Drift Detection"]
    continual["Continual-Learning Decision"]
    graph["Graph Flow"]

    data --> train --> validate --> approve --> demo --> drift --> continual --> graph
    validate -->|fail| train
    drift -->|breach| train
    demo -->|fail| approve
```

The HTML file is intentionally local-first. Later UI options can consume the same `LifecycleGraph` contract instead of owning lifecycle logic.
