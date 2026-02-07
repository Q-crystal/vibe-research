import pytest

pytest.importorskip("ultralytics")
pytest.importorskip("torch_pruning")

from vibe_research.recipes.yolov8_prune_backbone import prune_yolov8_backbone_from_yaml


def test_yolov8_backbone_prune_smoke_cpu():
    # Build from YAML to avoid downloading weights
    pruned = prune_yolov8_backbone_from_yaml("yolov8n.yaml", ratio=0.005, imgsz=128, device="cpu")
    assert pruned is not None
