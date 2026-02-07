from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vibe-research")
    p.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    return p

def main(args: list[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(args)
    return 0
