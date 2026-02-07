from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

import torch
import torch.nn as nn


class MissingDependencyError(RuntimeError):
    """Raised when optional deps for structured pruning are not installed."""


def _require_torch_pruning():
    try:
        import torch_pruning as tp  # type: ignore
    except Exception as e:  # pragma: no cover
        raise MissingDependencyError(
            "torch_pruning is required for DepGraph-based structured pruning. "
            "Install it (e.g., `pip install torch-pruning`) or add an extra in pyproject."
        ) from e
    return tp


LayerSelector = Callable[[str, nn.Module], bool]


@dataclass(frozen=True)
class DepGraphPruner:
    """Framework-agnostic wrapper around Torch-Pruning DepGraph.

    - Keep this kernel independent from Ultralytics.
    - YOLOv8 integration (backbone scoping, head ignore rules) will be in a separate module.
    """

    model: nn.Module
    example_inputs: Sequence[torch.Tensor]
    ignored_modules: tuple[nn.Module, ...] = ()
    layer_selector: Optional[LayerSelector] = None

    def prune_by_ratio(self, ratio: float) -> nn.Module:
        """Global structured pruning by ratio using per-filter L2 norm on Conv2d out-channels."""
        if not (0.0 < ratio < 1.0):
            raise ValueError(f"ratio must be in (0, 1), got {ratio}")

        tp = _require_torch_pruning()

        dg = tp.DependencyGraph().build_dependency(
            self.model,
            example_inputs=tuple(self.example_inputs),
        )

        prunable: list[tuple[str, nn.Conv2d]] = []
        for name, m in self.model.named_modules():
            if not isinstance(m, nn.Conv2d):
                continue
            if m in self.ignored_modules:
                continue
            if self.layer_selector is not None and not self.layer_selector(name, m):
                continue
            if m.out_channels <= 1:
                continue
            prunable.append((name, m))

        if not prunable:
            return self.model

        scored: list[tuple[float, nn.Conv2d, int]] = []
        for _, conv in prunable:
            w = conv.weight.detach()
            norms = torch.norm(w.reshape(w.shape[0], -1), p=2, dim=1)
            for i, s in enumerate(norms.tolist()):
                scored.append((float(s), conv, int(i)))

        scored.sort(key=lambda x: x[0])  # small -> prune first
        k = int(len(scored) * ratio)
        if k <= 0:
            return self.model

        per_layer: dict[nn.Conv2d, list[int]] = {}
        for _, conv, idx in scored[:k]:
            per_layer.setdefault(conv, []).append(idx)

        for conv, idxs in per_layer.items():
            idxs = sorted(set(idxs))
            group = dg.get_pruning_group(conv, tp.prune_conv_out_channels, idxs=idxs)
            if dg.check_pruning_group(group):
                group.prune()

        return self.model


def build_depgraph_pruner(
    model: nn.Module,
    example_inputs: Sequence[torch.Tensor],
    *,
    ignored_modules: Optional[Iterable[nn.Module]] = None,
    layer_selector: Optional[LayerSelector] = None,
) -> DepGraphPruner:
    return DepGraphPruner(
        model=model,
        example_inputs=tuple(example_inputs),
        ignored_modules=tuple(ignored_modules or ()),
        layer_selector=layer_selector,
    )
