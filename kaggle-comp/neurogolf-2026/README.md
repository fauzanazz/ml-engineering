# NeuroGolf 2026

Workspace for Kaggle competition [The 2026 NeuroGolf Championship](https://www.kaggle.com/competitions/neurogolf-2026).

## Objective

Build the smallest possible ONNX neural networks that solve ARC-AGI image transformation tasks.

Each task gives example input/output grids. A submitted network must exactly transform every input grid into its expected output grid. Correctness is required first; smaller networks score better.

Per-task score:

```text
score = max(1, 25 - ln(cost))
cost = total_parameters + memory_footprint_bytes
```

## Data

Dataset is extracted under `data/`.

```text
data/
  neurogolf_utils/
    neurogolf_utils.py
  task001.json
  task002.json
  ...
  task400.json
neurogolf-2026.zip
```

Summary from local archive:

- `400` task JSON files
- `1302` `train` examples
- `416` `test` examples
- `100000` `arc-gen` examples

Each task file contains:

- `train`: ARC-AGI-1 training examples
- `test`: ARC-AGI-1 test examples
- `arc-gen`: ARC-GEN-100K additional examples

Each grid is a rectangular matrix of integers `0..9`. Inputs are converted to tensors shaped `[1, 10, 30, 30]` using one-hot color encoding and zero-hot padding.

## Submission Format

Create `submission.zip` containing at most one ONNX file per task:

```text
task001.onnx
task002.onnx
...
task400.onnx
```

## ONNX Constraints

- Max file size: `1.44MB`
- Static tensor and parameter shapes required
- One graph input and one graph output expected
- Disallowed ops include `Loop`, `Scan`, `NonZero`, `Unique`, `Script`, `Function`, `Compress`
- Sequence operators are rejected
- Custom domains, functions, and subgraphs are rejected by validator

## Utility Helpers

Official helpers live in `data/neurogolf_utils/neurogolf_utils.py`.

Common functions:

- `load_examples(task_num)`
- `convert_to_numpy(example)`
- `convert_from_numpy(benchmark)`
- `single_layer_conv2d_network(weight_fn, kernel_size)`
- `verify_network(network, task_num, examples)`
- `score_network(sanitized, trace_path)`

Example workflow:

```python
from data.neurogolf_utils import neurogolf_utils

examples = neurogolf_utils.load_examples(1)
# build ONNX model for task 001
# neurogolf_utils.verify_network(network, 1, examples)
```

## Kaggle Commands

List files:

```bash
kaggle competitions files neurogolf-2026 --page-size 500
```

Read competition page content:

```bash
kaggle competitions pages neurogolf-2026 --content --page-name Evaluation
```

Submit:

```bash
kaggle competitions submit -c neurogolf-2026 -f submission.zip -m "message"
```

## Timeline

All deadlines are `23:59 UTC`.

- April 15, 2026: Start Date
- July 8, 2026: Entry Deadline
- July 8, 2026: Team Merger Deadline
- July 15, 2026: Final Submission Deadline

## Prizes

Total prizes: `$50,000`.

- First Prize: `$12,000`
- Second Prize: `$10,000`
- Third Prize: `$10,000`
- Top Student Team: `$8,000`
- Longest Leader: `$10,000`
