# ADR 0002: Use DepGraph-based structured pruning as baseline kernel

## Status
Accepted

## Context
YOLOv8 backbone includes multi-branch connections (concat/skip). Manual channel propagation is error-prone.

## Decision
Adopt Torch-Pruning DependencyGraph (DepGraph) style pruning plan generation as baseline kernel.
Keep the kernel framework-agnostic; add Ultralytics/YOLOv8 integration in a separate wrapper module.

## Consequences
- Pros: robust dependency handling; faster iteration for new pruning strategies.
- Cons: requires example inputs for tracing; env needs torch-pruning for execution (readable error otherwise).
