from __future__ import annotations

import torch
import torch.nn as nn

from vibe_research.integrations.ultralytics import (
    build_yolov8_from_yaml,
    get_backbone_len_from_yaml,
    make_yolov8_backbone_selector,
)
from vibe_research.pruning import build_depgraph_pruner


def prune_yolov8_backbone_from_yaml(
    model_yaml: str,
    *,
    ratio: float = 0.01,
    imgsz: int = 128,
    device: str = "cpu",
) -> nn.Module:
    """Smoke-level pruning: prune backbone-root conv channels by a small ratio."""
    model = build_yolov8_from_yaml(model_yaml).to(device)
    model.eval()

    x = torch.randn(1, 3, imgsz, imgsz, device=device)

    # infer backbone boundary
    backbone_len = get_backbone_len_from_yaml(model) or 0
    selector = make_yolov8_backbone_selector(backbone_len) if backbone_len > 0 else None

    pruner = build_depgraph_pruner(model, [x], layer_selector=selector)
    pruned = pruner.prune_by_ratio(ratio)

    # forward check
    with torch.no_grad():
        _ = pruned(x)

    return pruned
