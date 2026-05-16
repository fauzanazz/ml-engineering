# Step 23: GitHub Actions CI Workflow

## Goal

Add a GitHub Actions workflow that runs `production-patterns domain` validation on push and pull request.

## User Story

Sebagai ML engineer, gw tidak cuma punya local CI, tapi juga remote CI yang jalan otomatis saat push/PR supaya regressions ketahuan sebelum merge.

## Workflow

Workflow file:

```text
.github/workflows/ml-production-ecosystem-ci.yml
```

Triggers:

- `push`
- `pull_request`

Path scope:

- `ml-production-ecosystem/**`
- `.github/workflows/ml-production-ecosystem-ci.yml`

Main command:

```bash
cd ml-production-ecosystem
./scripts/validate-production-patterns.sh
```

Workflow setup:

- checkout repo
- setup Python `3.13`
- install `uv`
- run local validation script

## Documentation

`docs/domains/production-patterns/local-ci.md` notes that local CI mirrors GitHub Actions remote CI.

## Key Files

- `.github/workflows/ml-production-ecosystem-ci.yml`
- `scripts/validate-production-patterns.sh`
- `docs/domains/production-patterns/local-ci.md`
- `tests/test_local_ci.py`

## Pattern

```text
push or pull_request
  -> GitHub Actions workflow
  -> cd ml-production-ecosystem
  -> ./scripts/validate-production-patterns.sh
  -> focused production-pattern validation
```

## Out Of Scope

- Full monorepo CI.
- Docker build CI.
- Deployment to environment.
- Secrets.
- Coverage upload.
- Matrix builds.

## Acceptance Criteria

- Workflow YAML exists and parses.
- Workflow calls `./scripts/validate-production-patterns.sh`.
- Workflow scoped to `ml-production-ecosystem` changes.
- Docs mention local CI mirrors GitHub Actions.
- Tests assert workflow has triggers and script command.
- Existing tests stay green.

## Definition Of Done

`production-patterns domain` has local CI plus remote CI skeleton. Project covers train → gate → activate → serve → monitor → alert → rollback → deploy metadata → local CI → remote CI.

## Next Step

[Step 24](./step-24-production-compose-profile.md) adds Docker Compose production profile.
