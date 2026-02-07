# vibe-research Knowledge Base

This repo uses **docs-as-code**:

- `docs/cards/` — Translation Cards (input → ICD → variants → tests → repo placement)
- `docs/recipes/` — Keyword Recipes (术语/方法的一页配方)
- `docs/adr/` — Architecture Decision Records
- `docs/experiments/` — Reproduction guides
- `kb/glossary.yaml` — keyword → meaning / default_api / related_docs
- `kb/registry.yaml` — feature → artifacts paths

## Quickstart

```bash
python -m pip install -e ".[dev]"
pytest -q
ruff check .
ruff format --check .
```
