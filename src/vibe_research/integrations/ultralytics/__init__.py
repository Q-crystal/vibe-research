"""Ultralytics integration layer (optional dependency)."""

from .yolov8 import (
    build_yolov8_from_yaml,
    get_backbone_len_from_yaml,
    make_yolov8_backbone_selector,
)

__all__ = ["build_yolov8_from_yaml", "get_backbone_len_from_yaml", "make_yolov8_backbone_selector"]
