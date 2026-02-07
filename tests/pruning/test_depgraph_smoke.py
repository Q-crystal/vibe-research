import pytest
import torch
import torch.nn as nn

from vibe_research.pruning.depgraph import MissingDependencyError, build_depgraph_pruner


def test_missing_torch_pruning_error_is_readable():
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, padding=1),
    )
    x = torch.randn(1, 3, 16, 16)

    pruner = build_depgraph_pruner(model, [x])

    try:
        import torch_pruning  # noqa: F401
    except Exception:
        with pytest.raises(MissingDependencyError) as ei:
            pruner.prune_by_ratio(0.2)
        assert "torch_pruning is required" in str(ei.value)
    else:
        pruned = pruner.prune_by_ratio(0.2)
        y = pruned(x)
        assert y.shape[0] == 1
