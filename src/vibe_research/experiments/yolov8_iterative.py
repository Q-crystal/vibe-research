from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from vibe_research.integrations.ultralytics import build_yolov8_from_yaml


@dataclass(frozen=True)
class IterCfg:
    model_yaml: str = "yolov8n.yaml"
    steps: int = 5
    total_ch_sparsity: float = 0.5
    imgsz: int = 640
    device: str = "cuda:0"
    out_dir: str = "runs/yolov8_iterative"


def _params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def _latency(m: nn.Module, x: torch.Tensor, warmup: int = 5, iters: int = 20) -> float:
    m.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = m(x)
        if x.is_cuda:
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                _ = m(x)
            end.record()
            torch.cuda.synchronize()
            ms = start.elapsed_time(end) / iters
            return float(ms)
        else:
            t0 = time.perf_counter()
            for _ in range(iters):
                _ = m(x)
            t1 = time.perf_counter()
            return float((t1 - t0) * 1000.0 / iters)


def _infer_shortcut(bottleneck) -> bool:
    """Infer if bottleneck has shortcut connection."""
    c1 = bottleneck.cv1.conv.in_channels
    c2 = bottleneck.cv2.conv.out_channels
    return c1 == c2 and hasattr(bottleneck, "add") and bottleneck.add


def _replace_c2f_with_c2f_v2(module: nn.Module) -> None:
    """
    Replace C2f with C2f_v2 to avoid channel mismatch during pruning.
    C2f uses chunk(2, 1) which requires even number of channels.
    C2f_v2 uses separate convolutions which is more pruning-friendly.
    """
    try:
        from ultralytics.nn.modules import C2f
    except ImportError:
        return  # ultralytics not installed

    for name, child in module.named_children():
        if isinstance(child, C2f):
            # Create C2f_v2 with same parameters
            shortcut = _infer_shortcut(child.m[0]) if len(child.m) > 0 else False
            c2f_v2 = C2f_v2(
                child.cv1.conv.in_channels,
                child.cv2.conv.out_channels,
                n=len(child.m),
                shortcut=shortcut,
                g=child.m[0].cv2.conv.groups if len(child.m) > 0 else 1,
                e=child.c / child.cv2.conv.out_channels,
            )
            _transfer_c2f_weights(child, c2f_v2)
            setattr(module, name, c2f_v2)
        else:
            _replace_c2f_with_c2f_v2(child)


class C2f_v2(nn.Module):
    """CSP Bottleneck with 2 convolutions - pruning-friendly version."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        from ultralytics.nn.modules import Bottleneck, Conv

        super().__init__()
        self.c = int(c2 * e)
        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


def _transfer_c2f_weights(c2f, c2f_v2) -> None:
    """Transfer weights from C2f to C2f_v2."""
    c2f_v2.cv2 = c2f.cv2
    c2f_v2.m = c2f.m

    state_dict = c2f.state_dict()
    state_dict_v2 = c2f_v2.state_dict()

    # Transfer cv1 weights (split into cv0 and cv1)
    old_weight = state_dict["cv1.conv.weight"]
    half_channels = old_weight.shape[0] // 2
    state_dict_v2["cv0.conv.weight"] = old_weight[:half_channels]
    state_dict_v2["cv1.conv.weight"] = old_weight[half_channels:]

    # Transfer batchnorm parameters
    for bn_key in ["weight", "bias", "running_mean", "running_var"]:
        old_bn = state_dict[f"cv1.bn.{bn_key}"]
        state_dict_v2[f"cv0.bn.{bn_key}"] = old_bn[:half_channels]
        state_dict_v2[f"cv1.bn.{bn_key}"] = old_bn[half_channels:]

    # Transfer remaining weights
    for key in state_dict:
        if not key.startswith("cv1."):
            state_dict_v2[key] = state_dict[key]

    # Transfer non-method attributes
    for attr_name in dir(c2f):
        attr_value = getattr(c2f, attr_name)
        if not callable(attr_value) and "_" not in attr_name:
            setattr(c2f_v2, attr_name, attr_value)

    c2f_v2.load_state_dict(state_dict_v2)


def run_iterative_pruning(cfg_dict: dict[str, Any]) -> dict[str, Any]:
    """Iteratively prune YOLOv8 with torch-pruning GroupNormPruner."""
    try:
        import torch_pruning as tp  # type: ignore
        from ultralytics.nn.modules import Detect
    except Exception as e:
        raise RuntimeError(f"Required dependencies not available: {e}") from e

    cfg = IterCfg(**cfg_dict)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.device)
    model = build_yolov8_from_yaml(cfg.model_yaml).to(device).eval()
    x = torch.randn(1, 3, cfg.imgsz, cfg.imgsz, device=device)

    # Replace C2f with C2f_v2 for pruning compatibility
    _replace_c2f_with_c2f_v2(model)
    model = model.to(device).eval()  # Ensure all new modules are on correct device

    # Calculate pruning ratio per step
    pruning_ratio = 1 - math.pow((1 - cfg.total_ch_sparsity), 1 / cfg.steps)

    records: list[dict[str, Any]] = []
    base_params = _params(model)

    for step in range(cfg.steps):
        rec: dict[str, Any] = {"step": step, "status": "ok"}
        rec["params_before"] = _params(model)

        try:
            # Prepare ignored layers (Detect head should not be pruned)
            ignored_layers = []
            for m in model.modules():
                if isinstance(m, Detect):
                    ignored_layers.append(m)

            # Create pruner for this step
            pruner = tp.pruner.GroupNormPruner(
                model,
                example_inputs=x,
                importance=tp.importance.GroupMagnitudeImportance(),
                iterative_steps=1,
                pruning_ratio=pruning_ratio,
                ignored_layers=ignored_layers,
            )

            # Execute pruning
            pruner.step()

            # Verify forward pass works
            model.eval()
            with torch.no_grad():
                _ = model(x)

            rec["latency_ms"] = _latency(model, x)
            rec["params_after"] = _params(model)
            rec["params_reduction_pct"] = (1 - rec["params_after"] / base_params) * 100

            # Save pruned model
            torch.save(model, out / f"model_step{step}.pt")

            # Clean up pruner
            del pruner

        except Exception as e:
            rec["status"] = "failed"
            rec["error"] = repr(e)
            (out / "failure.json").write_text(
                json.dumps(rec, indent=2, ensure_ascii=False)
            )
            records.append(rec)
            break

        records.append(rec)

    summary = {"cfg": cfg_dict, "records": records}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary
