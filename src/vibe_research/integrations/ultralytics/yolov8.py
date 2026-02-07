from __future__ import annotations

import re
from typing import Callable, Optional

import torch.nn as nn

LayerSelector = Callable[[str, nn.Module], bool]


def build_yolov8_from_yaml(model_yaml: str) -> nn.Module:
    """Build a YOLOv8 model from YAML without requiring weights."""
    from ultralytics import YOLO  # optional dependency

    y = YOLO(model_yaml)
    return y.model  # BaseModel (nn.Module)


def get_backbone_len_from_yaml(model: nn.Module) -> Optional[int]:
    """Infer backbone length (number of top-level layers) from model.yaml dict if available."""
    y = getattr(model, "yaml", None)
    if isinstance(y, dict) and "backbone" in y and isinstance(y["backbone"], list):
        return len(y["backbone"])
    return None


_TOP_RE = re.compile(r"^model\.(\d+)(?:\.|$)")


def make_yolov8_backbone_selector(backbone_len: int) -> LayerSelector:
    """Select modules whose top-level index is within backbone (i < backbone_len)."""

    def _sel(name: str, module: nn.Module) -> bool:
        m = _TOP_RE.match(name)
        if not m:
            return False
        idx = int(m.group(1))
        return idx < backbone_len

    return _sel
