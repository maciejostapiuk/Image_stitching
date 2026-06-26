#!/usr/bin/env python3
"""
07_stitch.py
============
STAGE 7: assemble the final mosaic from the matched transforms.

Produces:
    - a pyramidal OME-TIFF (memory-safe; multi-resolution, viewer-friendly)
    - the stitch layout JSON (where every tile was placed)

The flat PNG output was removed: on large mosaics it duplicated the render
work and slowed the stage down. The pyramidal OME-TIFF already contains a
full-resolution level plus downsampled overviews and is the better deliverable.

Placement uses the "good" connections (metadata-scaled BFS), then fills
all-failed tiles from their neighbors' positions so no tile is dropped.

Inputs
------
    output/stitch_graph.json
    output/transforms_translation.json
    the tile images

Outputs
-------
    output/stitched_pyramid.ome.tif
    output/stitch_layout.json

Run
---
    python scripts/07_stitch.py
    PYRAMID_COMPRESSION=jpeg python scripts/07_stitch.py   # smaller, lossy
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import common
from common import log, section, save_json, load_json, load_tiles, load_tiles_from_graph

import stitcher


def main() -> None:
    section("STAGE 7  -  STITCH (pyramidal OME-TIFF)")
    config.validate_dataset()

    if not config.GRAPH_PATH.exists():
        raise FileNotFoundError(
            f"Missing {config.GRAPH_PATH}. Run 01_build_borders.py first.")
    if not config.TRANSFORMS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {config.TRANSFORMS_PATH}. Run 03_match.py first.")

    graph = load_json(config.GRAPH_PATH)
    results = load_json(config.TRANSFORMS_PATH)

    # tile list: prefer the real folder, fall back to the graph
    try:
        tiles = load_tiles(config.input_folder())
    except FileNotFoundError:
        tiles = []
    if not tiles:
        log("Input folder unavailable; reconstructing tiles from graph.")
        tiles = load_tiles_from_graph(graph)
    log(f"Tiles: {len(tiles)}  |  connections: {len(graph)}")

    tile_shapes = stitcher.get_tile_shapes(tiles)
    _summary, all_failed, no_conn = stitcher.summarize_statuses(
        tiles, graph, results)
    log(f"All-failed tiles: {len(all_failed)}  |  no-connection: {len(no_conn)}")

    log("\nBuilding layout from good connections...")
    layout, sx, sy = stitcher.build_stitch_layout(tiles, results, tile_shapes)
    log(f"  metadata scale: x={sx:.4f} y={sy:.4f}")

    fallback = stitcher.neighbor_average_fallback(
        tiles, graph, layout, all_failed, no_conn, sx, sy)

    omit = all_failed if config.OMIT_ALL_FAILED_TILES_FROM_STITCH else set()

    log("\nWriting pyramidal OME-TIFF...")
    pyramid_info = stitcher.write_pyramidal_ome_tiff(
        layout, config.PYRAMID_OME_TIFF_PATH, omitted_tile_ids=omit)

    layout_json = {
        "dataset": config.DATASET_NAME,
        "source_graph": str(config.GRAPH_PATH),
        "source_transforms": str(config.TRANSFORMS_PATH),
        "match_method": config.MATCH_METHOD,
        "blend_mode": config.BLEND_MODE,
        "used_statuses_for_layout": sorted(config.STITCH_USABLE_STATUSES),
        "metadata_scale_x": sx, "metadata_scale_y": sy,
        "pyramid_info": pyramid_info,
        "all_failed_tile_ids": sorted(int(t) for t in all_failed),
        "no_connection_tile_ids": sorted(int(t) for t in no_conn),
        "all_failed_tile_neighbor_average_fallback": fallback,
        "tiles": layout,
    }
    save_json(layout_json, config.STITCH_LAYOUT_PATH)

    section("STITCH SUMMARY")
    log(f"OME-TIFF:   {config.PYRAMID_OME_TIFF_PATH}")
    log(f"Layout:     {config.STITCH_LAYOUT_PATH}")
    log(f"Canvas:     {pyramid_info['full_width']} x "
        f"{pyramid_info['full_height']}  |  "
        f"{len(pyramid_info['levels'])} pyramid level(s)  |  "
        f"blend={config.BLEND_MODE}")
    log("\nStage 7 complete.")


if __name__ == "__main__":
    main()
