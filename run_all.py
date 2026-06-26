#!/usr/bin/env python3
"""
run_all.py
==========
Orchestrator: run the pipeline stages in order.

Currently wired stages
----------------------
    01_build_borders.py   - discover tiles, build the borders graph
    02_cut_borders.py     - cut the edge strips to be matched

Later stages (matching, metrics, tree, global optimization, stitching,
error estimation) plug in here as they are added, keeping the same ordered
numbering.

Usage
-----
    python run_all.py                 # run all wired stages
    python run_all.py --from 2        # start at stage 2
    python run_all.py --only 1        # run just stage 1
    DATASET_NAME=histology-large python run_all.py

The orchestrator simply shells out to each numbered script in scripts/, so
each stage stays independently runnable and debuggable on its own.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "scripts"


def discover_stages() -> list[tuple[int, Path]]:
    """Find numbered stage scripts (NN_*.py) and return them sorted."""
    stages = []
    for p in sorted(SCRIPTS_DIR.glob("[0-9][0-9]_*.py")):
        try:
            num = int(p.name[:2])
        except ValueError:
            continue
        stages.append((num, p))
    return stages


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stitching pipeline stages.")
    parser.add_argument("--from", dest="start", type=int, default=0,
                        help="start at this stage number (inclusive)")
    parser.add_argument("--to", dest="end", type=int, default=99,
                        help="stop after this stage number (inclusive)")
    parser.add_argument("--only", dest="only", type=int, default=None,
                        help="run only this single stage")
    args = parser.parse_args()

    stages = discover_stages()
    if not stages:
        print("No stage scripts found in scripts/. Nothing to run.")
        sys.exit(1)

    if args.only is not None:
        stages = [(n, p) for n, p in stages if n == args.only]
    else:
        stages = [(n, p) for n, p in stages if args.start <= n <= args.end]

    if not stages:
        print("No stages match the requested range.")
        sys.exit(1)

    print(f"Pipeline: running {len(stages)} stage(s): "
          f"{[n for n, _ in stages]}")

    for num, path in stages:
        print(f"\n>>> STAGE {num}: {path.name}")
        result = subprocess.run([sys.executable, str(path)])
        if result.returncode != 0:
            print(f"\n!!! Stage {num} ({path.name}) failed "
                  f"(exit {result.returncode}). Stopping.")
            sys.exit(result.returncode)

    print("\nAll requested stages completed.")


if __name__ == "__main__":
    main()
