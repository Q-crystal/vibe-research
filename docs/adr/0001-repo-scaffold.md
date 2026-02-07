# ADR 0001: Repository scaffold (docs-as-code + registries)

## Status
Accepted

## Context
We want a repo that can reliably translate research ideas → ICD → runnable code + tests, and persist key decisions into Markdown.

## Decision
Adopt:
- docs-as-code (docs/, kb/)
- standard Python layout (src/, tests/)
- Ruff + pytest + pre-commit + CI

## Consequences
- Knowledge and decisions are versioned with code.
- Changes are auditable via git diff and CI.
