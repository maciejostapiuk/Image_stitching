#!/usr/bin/env python3
"""
03_match.py
===========
STAGE 3: compute the per-border translation between adjacent tiles.

The MATCHER METHOD is user-selectable:
    sift   - classic SIFT keypoints + ratio test + translation RANSAC (CPU, fast)
    roma   - RoMaV2 dense neural matcher + translation RANSAC (needs GPU + romav2)
    hybrid - SIFT first; fall back to RoMa on borders SIFT isn't confident about

Both methods produce the IDENTICAL output schema, so everything downstream
(metrics, spanning tree, global optimization, stitching) works the same
regardless of which matcher you pick.

For every connection in stitch_graph.json this records dx, dy (the measured
shift), an inlier_ratio quality score, and a good/suspicious/failed status.

Output
------
    output/transforms_translation.json
    output/debug_matches/<connection>.png        (match visualizations)

Run
---
    python scripts/03_match.py                    # uses config.MATCH_METHOD (default sift)
    MATCH_METHOD=sift python scripts/03_match.py
    MATCH_METHOD=roma python scripts/03_match.py    # requires GPU + romav2
    MATCH_METHOD=hybrid python scripts/03_match.py  # SIFT, with RoMa fallback
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import common
from common import log, section, save_json, load_json

import matchers


def main() -> None:
    section("STAGE 3  -  FEATURE MATCHING")
    config.validate_dataset()

    method = config.MATCH_METHOD.lower()
    if method not in config.VALID_MATCH_METHODS:
        raise ValueError(
            f"Unknown MATCH_METHOD {method!r}. "
            f"Choose one of: {sorted(config.VALID_MATCH_METHODS)}")
    log(f"Match method: {method}")
    log(f"Edge strip:   {config.EDGE_STRIP} px")

    if not config.GRAPH_PATH.exists():
        raise FileNotFoundError(
            f"Missing {config.GRAPH_PATH}. Run 01_build_borders.py first.")

    graph = load_json(config.GRAPH_PATH)
    log(f"Connections in graph: {len(graph)}")

    config.DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    matcher = matchers.get_matcher(method)

    results = {}
    counts = {"good": 0, "suspicious": 0, "failed": 0}
    method_counts = {}    # for hybrid: how many borders each backend handled
    for name, conn in graph.items():
        result = matcher(name, conn, edge_strip=config.EDGE_STRIP,
                         debug_dir=config.DEBUG_DIR)
        results[name] = result
        st = result.get("status", "failed")
        counts[st] = counts.get(st, 0) + 1
        used = result.get("method", method)
        method_counts[used] = method_counts.get(used, 0) + 1
        dxy = (f"dx={result['dx']:.1f} dy={result['dy']:.1f}"
               if "dx" in result else f"({result.get('reason')})")
        via = f"  via {used}" if method == "hybrid" else ""
        log(f"  {name:24s} {st:11s} {dxy}{via}")

    save_json(results, config.TRANSFORMS_PATH)

    section("MATCHING SUMMARY")
    log(f"Dataset:     {config.DATASET_NAME}")
    log(f"Method:      {method}")
    log(f"Connections: {len(graph)}")
    log(f"  good:       {counts['good']}")
    log(f"  suspicious: {counts['suspicious']}")
    log(f"  failed:     {counts['failed']}")
    if method == "hybrid":
        log("  by backend:")
        for backend, n in sorted(method_counts.items()):
            log(f"    {backend:6s}: {n}")
    log("\nStage 3 complete.")


if __name__ == "__main__":
    main()
