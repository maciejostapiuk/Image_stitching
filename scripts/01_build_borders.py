#!/usr/bin/env python3
"""
01_build_borders.py
===================
STAGE 1 of the stitching pipeline: discover tiles and build the BORDERS graph.

What it does
------------
1. Scans the dataset's Images-FOV folder for tiles named <id>_<x>_<y>.bmp.
2. Places tiles on an integer grid by their stage (x, y) coordinates.
3. For each tile, creates a connection to its RIGHT and BOTTOM neighbor
   (so every shared border is represented exactly once).
4. Records, per connection, which edge of each tile forms the shared border
   ("right<->left" or "bottom<->top").

Output
------
    output/stitch_graph.json

This file is the backbone of the whole pipeline: every later stage keys its
results by the same connection names ("connection_<a>_<b>").

Run
---
    python scripts/01_build_borders.py
    DATASET_NAME=histology-large python scripts/01_build_borders.py
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

# allow "import config / common" when run from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import common
from common import log, section, save_json, load_tiles


def build_graph(tiles: list[dict]) -> "OrderedDict":
    """Build the connection graph from a tile list.

    Tiles are arranged on a grid by sorting their unique x and y stage
    coordinates. Each tile links to the neighbor immediately to its right
    and immediately below, if present.
    """
    if not tiles:
        return OrderedDict()

    xs = sorted({t["x"] for t in tiles})
    ys = sorted({t["y"] for t in tiles})
    coord_to_tile = {(t["x"], t["y"]): t for t in tiles}

    # iterate row-major for stable, readable connection ordering
    sorted_tiles = sorted(tiles, key=lambda t: (ys.index(t["y"]), xs.index(t["x"])))

    graph: "OrderedDict" = OrderedDict()
    for tile in sorted_tiles:
        col = xs.index(tile["x"])
        row = ys.index(tile["y"])

        # (edge_on_this_tile, neighbor_col, neighbor_row)
        candidates = [
            ("right", col + 1, row),
            ("bottom", col, row + 1),
        ]

        for edge, ncol, nrow in candidates:
            if ncol >= len(xs) or nrow >= len(ys):
                continue
            neighbor = coord_to_tile.get((xs[ncol], ys[nrow]))
            if neighbor is None:
                continue

            opposite = config.OPPOSITE_EDGE[edge]
            key = f"connection_{tile['id']}_{neighbor['id']}"
            graph[key] = {
                "label": (
                    f"slide_{tile['id']}:{edge} <-> "
                    f"slide_{neighbor['id']}:{opposite}"
                ),
                "type": "vertical" if edge == "right" else "horizontal",
                "a": {
                    "slide": tile["id"],
                    "edge": edge,
                    "file": tile["file"],
                    "coord": [tile["x"], tile["y"]],
                },
                "b": {
                    "slide": neighbor["id"],
                    "edge": opposite,
                    "file": neighbor["file"],
                    "coord": [neighbor["x"], neighbor["y"]],
                },
            }

    return graph


def main() -> None:
    section("STAGE 1  -  BUILD BORDERS GRAPH")
    config.validate_dataset()

    folder = config.input_folder()
    log(f"Dataset:      {config.DATASET_NAME}")
    log(f"Input folder: {folder}")
    log(f"Output dir:   {config.OUTPUT_DIR}")

    tiles = load_tiles(folder)
    log(f"\nTiles found: {len(tiles)}")
    if not tiles:
        raise RuntimeError(f"No tiles found in: {folder}")

    graph = build_graph(tiles)
    log(f"Connections found: {len(graph)}")

    # quick connection-type breakdown
    n_vert = sum(1 for c in graph.values() if c["type"] == "vertical")
    n_horiz = sum(1 for c in graph.values() if c["type"] == "horizontal")
    log(f"  vertical (left-right) borders: {n_vert}")
    log(f"  horizontal (top-bottom) borders: {n_horiz}")

    save_json(graph, config.GRAPH_PATH)
    log("\nStage 1 complete.")


if __name__ == "__main__":
    main()
