# YOLOv8 Backbone Pruning（Ultralytics 对接）

## 目标
在不依赖权重下载的前提下，通过 `YOLO("yolov8n.yaml")` 从 YAML 初始化模型，并对 backbone-root 的通道做小比例结构化剪枝（smoke）。

## 入口
- `vibe_research.recipes.yolov8_prune_backbone.prune_yolov8_backbone_from_yaml`

## 关键点
- backbone/head 边界来自 YAML 的 `backbone` 段长度（用于选择 root 层）。
- 剪枝依赖 DepGraph，避免 concat/skip 结构下手工传播通道。
