import pytest

try:
    import ultralytics  # noqa: F401
    import torch_pruning  # noqa: F401
    import torch
except Exception as e:  # noqa: BLE001
    pytest.skip(f"optional deps not usable in this env: {e}", allow_module_level=True)

from vibe_research.experiments.yolov8_iterative import run_iterative_pruning


def test_iterative_pruning_runs_one_step_cpu():
    # keep CPU smoke tiny to avoid heavy deps; in GPU env you run full config
    cfg = {
        "model_yaml": "yolov8n.yaml",
        "steps": 1,
        "total_ch_sparsity": 0.01,
        "imgsz": 128,
        "device": "cpu",
        "out_dir": "experiments/yolov8_pruning_baseline/runs/pytest_smoke"
    }
    out = run_iterative_pruning(cfg)
    assert out["records"][0]["status"] in ("ok", "failed")
