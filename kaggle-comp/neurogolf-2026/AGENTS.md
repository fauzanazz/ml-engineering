# Repository Instructions

## Clean Code Philosophy
Every line is a liability. Best code: no code.

- DRY: write twice = doing it wrong
- YAGNI: build only what is needed now
- KISS: complexity kills maintainability
- Readability > cleverness
- Self-documenting: names explain what; comments explain why
- Always use descriptive names, including in `map`/`filter`
- Prefer pure functions
- Use classes only for shared mutable state or real polymorphism
- Prefer composition over inheritance
- Define interfaces/types first; use classes only when behavior and state need runtime instances
- Flag modules over 200 LOC or classes over 5 public methods as god objects to split
- Functional chains may be short; avoid nested point-free puzzles

## Project Context
This repository is for Kaggle competition `neurogolf-2026`: The 2026 NeuroGolf Championship.

Competition URL: https://www.kaggle.com/competitions/neurogolf-2026

Goal: create one ONNX neural network per ARC-AGI task that exactly transforms input grids into output grids while minimizing model cost.

Cost formula per correct task:

```text
score = max(1, 25 - ln(cost))
cost = total_parameters + memory_footprint_bytes
```

## Local Data Layout
Dataset files are stored under `data/`.

- `data/task001.json` through `data/task400.json`: task definitions
- `data/neurogolf_utils/neurogolf_utils.py`: official utility and validator helpers
- `neurogolf-2026.zip`: original downloaded Kaggle archive

Each task JSON has fields:

- `train`: ARC-AGI-1 training examples
- `test`: ARC-AGI-1 test examples
- `arc-gen`: ARC-GEN-100K extra examples

Each example has:

- `input`: grid matrix
- `output`: grid matrix

Grid values are integers `0..9`. Networks receive tensors shaped `[1, 10, 30, 30]` with one-hot color channels and zero-hot padding outside original grid borders.

## Submission Rules
Submit `submission.zip` containing at most one ONNX file per task:

```text
task001.onnx
task002.onnx
...
task400.onnx
```

Important constraints:

- ONNX file size must be at most `1.44MB`
- All tensors and parameters must have statically defined shapes
- Graph should have one input and one output
- Custom domains, functions, and subgraphs are rejected by validator
- Disallowed op types include `Loop`, `Scan`, `NonZero`, `Unique`, `Script`, `Function`, `Compress`
- Sequence operators are rejected
- Tensor/node names containing `kernel_time` are rejected

## Useful Utility Functions
Use `data/neurogolf_utils/neurogolf_utils.py`.

Key functions:

- `load_examples(task_num)`
- `convert_to_numpy(example)`
- `convert_from_numpy(benchmark)`
- `single_layer_conv2d_network(weight_fn, kernel_size)`
- `verify_network(network, task_num, examples)`
- `score_network(sanitized, trace_path)`

## Kaggle Commands
List competition files:

```bash
kaggle competitions files neurogolf-2026 --page-size 500
```

List competition pages:

```bash
kaggle competitions pages neurogolf-2026
```

Read a page:

```bash
kaggle competitions pages neurogolf-2026 --content --page-name Evaluation
```

Download data:

```bash
kaggle competitions download -c neurogolf-2026
```

Submit:

```bash
kaggle competitions submit -c neurogolf-2026 -f submission.zip -m "message"
```

## Competition Timeline
All deadlines are `23:59 UTC`.

- Start Date: April 15, 2026
- Entry Deadline: July 8, 2026
- Team Merger Deadline: July 8, 2026
- Final Submission Deadline: July 15, 2026

## Implementation Guidance
- Keep generators small and task-specific unless abstraction clearly removes duplication.
- Validate each produced ONNX with `verify_network` before packaging.
- Prefer exact, deterministic construction over learned training loops where possible.
- Do not add broad frameworks or heavy dependencies unless required.
- Do not modify dataset files unless explicitly asked.
