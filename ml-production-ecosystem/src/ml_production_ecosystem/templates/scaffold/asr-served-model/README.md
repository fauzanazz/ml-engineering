# {{project_name}}

ASR served-model scaffold with speech-to-text contract, WER/CER quality gate, FastAPI app, and smoke test.

## Commands

```bash
uv run pytest
uv run uvicorn {{package_name}}.api:app --reload
```

## Flow

1. Put local audio samples in `data/`.
2. Replace `{{package_name}}/transcribe.py` with your model call.
3. Write metrics JSON from evaluation.
4. Gate release with `configs/project.yaml` thresholds.
