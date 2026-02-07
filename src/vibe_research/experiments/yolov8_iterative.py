from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from vibe_research.integrations.ultralytics import (
    build_yolov8_from_yaml,
    get_backbone_len_from_yaml,
    make_yolov8_backbone_selector,
)


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


def run_iterative_pruning(cfg_dict: dict[str, Any]) -> dict[str, Any]:
    """Iteratively prune YOLOv8 with torch-pruning MagnitudePruner (optional dependency)."""
    try:
        import torch_pruning as tp  # type: ignore
    except Exception as e:
        raise RuntimeError(f"torch_pruning is required for iterative pruning: {e}") from e

    cfg = IterCfg(**cfg_dict)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.device)
    model = build_yolov8_from_yaml(cfg.model_yaml).to(device).eval()
    x = torch.randn(1, 3, cfg.imgsz, cfg.imgsz, device=device)

    backbone_len = get_backbone_len_from_yaml(model) or 0
    selector = make_yolov8_backbone_selector(backbone_len) if backbone_len > 0 else None

    # Collect "ignored layers" as pruning roots: everything NOT in backbone (root selection only)
    ignored_layers: list[nn.Module] = []
    if backbone_len > 0:
        for name, m in model.named_modules():
            # ignore as roots if top-level idx >= backbone_len
            if name.startswith("model."):
                try:
                    idx = int(name.split(".")[1])
                except Exception:
                    continue
                if idx >= backbone_len and isinstance(m, nn.Conv2d):
                    ignored_layers.append(m)

    imp = tp.importance.MagnitudeImportance(p=2)  # weight L2
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=(x,),
        importance=imp,
        iterative_steps=int(cfg.steps),
        ch_sparsity=float(cfg.total_ch_sparsity),
        ignored_layers=ignored_layers,
    )

    records: list[dict[str, Any]] = []
    for step in range(cfg.steps):
        rec: dict[str, Any] = {"step": step, "status": "ok"}
        rec["params_before"] = _params(model)

        try:
            # interactive=True returns groups; we prune only roots inside backbone if selector provided
            groups = pruner.step(interactive=True)

            for g in groups:
                root = g[0][0]  # (dep, idxs) like structure; root module is g[0][0].target in some versions
                # best-effort: allow pruning; selector is enforced mainly by ignored_layers and backbone_len
                if pruner.DG.check_pruning_group(g):  # avoid full prune
                    g.prune()

            # forward + latency sanity
            with torch.no_grad():
                _ = model(x)
            rec["latency_ms"] = _latency(model, x)
            rec["params_after"] = _params(model)

            torch.save(model, out / f"model_step{step}.pt")
        except Exception as e:
            rec["status"] = "failed"
            rec["error"] = repr(e)
            # write failure and stop
            (out / "failure.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False))
            records.append(rec)
            break

        records.append(rec)

    summary = {"cfg": cfg_dict, "records": records}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary
