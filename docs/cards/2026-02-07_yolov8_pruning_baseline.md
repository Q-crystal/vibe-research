# Translation Card: YOLOv8 backbone pruning baseline (Milestone 0)

## Intent
先在 vibe-research 内沉淀"结构化剪枝内核"（DepGraph），保持与训练框架解耦。

## ICD
- Core: `DepGraphPruner`（torch-pruning 可选依赖）
- Next: Ultralytics/YOLOv8 wrapper + experiments 套件（Milestone 1）

## Artifacts
- `src/vibe_research/pruning/depgraph.py`
- `tests/pruning/test_depgraph_smoke.py`
- `configs/pruning/yolov8_baseline.yaml`
- `docs/recipes/torch_pruning_depgraph.md`
- `docs/cards/2026-02-07_yolov8_pruning_baseline.md`
- `docs/adr/0002-pruning-baseline-depgraph.md`

## Acceptance
- `pytest -q`
- `ruff check .`
- `ruff format --check .`
