# Translation Card: YOLOv8 Iterative Pruning (Milestone 2)

## Intent
把 YOLOv8 backbone 结构化剪枝从 smoke 升级为可复现的 iterative pruning/评测套件。

## Key references
- Ultralytics YAML 结构（backbone/head 分段，层连接格式）
- Torch-Pruning DepGraph/Group 与 High-level Pruner（MagnitudePruner, iterative_steps/ch_sparsity）

## Artifacts
- src/vibe_research/experiments/yolov8_iterative.py
- experiments/yolov8_pruning_baseline/run_iterative.py
- configs/pruning/yolov8_iterative.json
- tests/experiments/test_iterative_pruning_smoke.py

## Acceptance
- CPU env: pytest -q 通过（集成可 skip）
- GPU env: 能生成 runs/.../metrics.json，并记录每步 params/latency/状态
