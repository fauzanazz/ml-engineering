# {{project_name}}

Template for teams that already have model-specific training logic and want a local-first, contract-safe bootstrap without touching core package internals.

## 5 common task bootstrap matrix

The scaffold can be created with one of these task profiles:

- `classification`: single-label output (`label`)
- `regression`: numeric value output (`value`)
- `object-detection`: list of boxes (`detections`)
- `segmentation`: dense mask output (`mask`)
- `text-generation`: generated text output (`text`)

Set `--task` when bootstrapping `existing-model-wrapper`:

```bash
uv run ml-struct new my-project \
  --preset existing-model-wrapper \
  --task classification
```

### Non-recommender bootstrap

Use `configs/project.yaml` to define the training seam.

```yaml
training:
  type: command
  command:
    - python
    - -m
    - {{package_name}}.train
    - --summary-path
    - reports/training-summary.json
  summary_path: reports/training-summary.json
```

For framework-tagged minima, keep command contract and set intent:

```yaml
training:
  type: onnx
  framework: onnx
  command:
    - python
    - -m
    - {{package_name}}.train
    - --summary-path
    - reports/onnx-training-summary.json
  summary_path: reports/onnx-training-summary.json

training:
  type: pytorch
  framework: pytorch
  command:
    - python
    - -m
    - {{package_name}}.train
    - --summary-path
    - reports/pytorch-training-summary.json
  summary_path: reports/pytorch-training-summary.json
```

`training.command` must create summary JSON with keys:
`model_name`, `version`, `artifact_uri`, `metrics_uri`.

A runnable example is scaffolded at `{{package_name}}/train.py`.
Use:

```bash
python -m {{package_name}}.train --summary-path reports/training-summary.json
```

The example now runs a tiny real optimization loop (`_run_mini_loop`) each run.
It writes `summary` + `metrics` files compatible with `quality_gate`:
- `reports/training-summary.json`
- `reports/metrics.json`

To bootstrap from a previous model artifact, pass `--model-state` with a JSON object:

```json
{"weights": [0.05, -0.02, 0.01, 0.03], "bias": 0.02}
```

The file is loaded as initial parameters before training.

## Contract outputs per task

Generated scaffolds wire model contract fields to the selected task:

- `task_type`
- `prediction_key`
- `input_schema_uri`
- `output_schema_uri`

These resolve to:

```text
schemas/<task>/input.json
schemas/<task>/output.json
```

Schema files are created during scaffold bootstrap so manifest validation can run immediately.

## Adapter seam

`{{package_name}}/adapter.py` is intentionally small and framework-neutral. Keep loader helpers next to your model code and return only contract-aligned summary dictionaries.

## Commands

```bash
uv run pytest
```

## Next

Replace placeholder logic with your model code. Keep contracts stable.
