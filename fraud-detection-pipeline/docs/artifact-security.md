# Artifact Security

## Trust boundary

`bundle.joblib` contains the fitted estimator and `FeaturePipeline`. Joblib uses Python pickle internally; loading it can execute arbitrary code. Only load a bundle produced by a trusted run in a trusted environment.

The scoring CLI requires an explicit acknowledgement:

```bash
uv run fraud-detect-score \
  --run-dir artifacts/runs/<run-id> \
  --input-path examples/transaction.json \
  --trust-artifact
```

The Python API applies the same boundary:

```python
bundle = load_bundle(run_dir, trusted=True)
```

Without `trusted=True`, loading fails with `Refusing to load pickle-based artifact without trusted=True`.

## Integrity verification

Schema-version-1 `config.json` records a SHA-256 manifest for `bundle.joblib`. `load_bundle` verifies that digest before calling `joblib.load`; a mismatch fails before deserialization. This detects accidental corruption or substitution relative to the trusted manifest.

SHA-256 is an integrity check, not a sandbox, signature, or proof that the original producer was trustworthy. A malicious actor able to replace both the bundle and manifest can still supply executable pickle content.

`evaluation.npz` has a separate SHA-256 manifest. Report generation verifies it and reads only held-out labels and scores; it never loads `bundle.joblib`.

## Repository policy

- Do not commit or distribute `bundle.joblib` from local runs.
- Do not commit `evaluation.npz`, raw CSV data, or row-level predictions.
- Publish the generated Markdown report and SVG plots instead.
- Preserve `config.json` and `metrics.json` locally when auditing a run, but treat them as run evidence rather than a trust signature.
- Recreate a bundle from locked source and a verified dataset when provenance is uncertain.

The legacy `model.txt`/`model.joblib` split described in early step documentation is no longer the current artifact contract. See the [README artifact contract](../README.md#artifact-contract-and-trust).
