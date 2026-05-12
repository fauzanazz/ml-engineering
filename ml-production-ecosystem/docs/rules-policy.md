# Rules Policy

This repository is public and CV-oriented. Private rules must not be committed.

## Private Rules Location

Use local-only path:

```text
ml-production-ecosystem/.rules/
```

This path is ignored by git.

## Allowed in Public Docs

- Learning goals
- Architecture notes
- Step-by-step implementation docs
- Tradeoffs and decisions safe for public viewing

## Not Allowed in Public Docs

- Private agent instructions
- Personal credentials or tokens
- Non-public infrastructure details
- Sensitive local workflow notes

## Workflow

1. Keep private guidance in `ml-production-ecosystem/.rules/` if needed.
2. Keep public learning documentation in tracked markdown files.
3. Before committing, run `git status --short` and confirm no `.rules/` files appear.
