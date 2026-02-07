from __future__ import annotations

import argparse
import json
from pathlib import Path

from vibe_research.experiments.yolov8_iterative import run_iterative_pruning


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to YAML/JSON config (JSON for now).")
    args = ap.parse_args()

    p = Path(args.config)
    cfg = json.loads(p.read_text())
    out = run_iterative_pruning(cfg)
    print(json.dumps(out["records"][-1], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
