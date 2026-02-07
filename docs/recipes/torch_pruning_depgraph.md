# Torch-Pruning DepGraph（结构化剪枝配方）

## 目的
用一次前向构建依赖图（Dependency Graph），自动生成"剪某层会连带影响哪些层"的 pruning plan，适合含 concat/skip 的结构。

## 本仓库最小接口
- `vibe_research.pruning.build_depgraph_pruner(model, example_inputs)`
- `DepGraphPruner.prune_by_ratio(ratio)`

## 关键前提
- 必须提供可运行的 `example_inputs` 来完成 tracing / dependency build。
- baseline 先从 **Conv2d out_channels** 结构化剪枝开始；YOLOv8 的 C2f/SPPF 等模块级策略在后续里程碑实现。

## 常见坑
- 不同输入 shape 可能触发不同 trace：建议固定输入形状。
- 剪后导出/融合：在与 Ultralytics 对接后再统一处理。
