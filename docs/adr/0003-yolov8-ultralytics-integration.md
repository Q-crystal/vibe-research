# ADR 0003: Add Ultralytics integration for YOLOv8 pruning smoke

## Status
Accepted

## Context
We need a concrete baseline target (YOLOv8) to validate DepGraph pruning kernel end-to-end.

## Decision
Add an optional integration layer that can build YOLOv8 from YAML (no weights) and provide a backbone selector
derived from YAML's backbone section. Keep tests import-skipped unless ultralytics + torch_pruning are installed.

## Consequences
- Pros: stable CI; clear boundary for later multimodal integration.
- Cons: first iteration is smoke-level (small ratio) to reduce coupling risk; full-scale pruning comes next milestone.
