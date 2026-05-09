# Artifact Security

## joblib model files (`model.joblib`)

`joblib.load` deserialises Python objects using `pickle` internally.
**Loading a `.joblib` file from an untrusted source can execute arbitrary code.**

Rules:
- Only load `model.joblib` files produced by this pipeline from a trusted source (your own CI, a verified artifact store).
- Never load a `.joblib` received over an unauthenticated channel.
- When a `.joblib` is written, `metrics.json` will contain a `model_artifact_warning` field as a reminder.

## LightGBM model files (`model.txt`)

LightGBM's native text format is not a deserialisation format and does not execute code on load.
No special trust requirement beyond data integrity.

## Stable artifact manifest

`config.json` for each run includes a `model_artifact` field with the filename (`model.txt` or `model.joblib`).
Consumers should read `config["model_artifact"]` to locate the model file rather than hard-coding the name.
