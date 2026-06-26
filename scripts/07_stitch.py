#!/usr/bin/env python3
"""
07_stitch.py
============
STAGE 7: write the pyramidal OME-TIFF from an existing stitch_layout.json.

This stage does NOT rebuild the layout.

Inputs:
    output/stitch_layout.json
    tile images referenced inside stitch_layout.json

Outputs:
    output/stitched_pyramid.ome.tif
    updated output/stitch_layout.json with final pyramid_info
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import common
from common import log, section, save_json, load_json

import stitcher


def main() -> None:
    section("STAGE 7  -  STITCH FROM GLOBAL LAYOUT JSON")
    config.validate_dataset()

    if not config.STITCH_LAYOUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing {config.STITCH_LAYOUT_PATH}. "
            f"Run scripts/06_global_layout.py first."
        )

    layout_data = load_json(config.STITCH_LAYOUT_PATH)

    if "tiles" not in layout_data:
        raise RuntimeError(
            f"{config.STITCH_LAYOUT_PATH} does not contain a 'tiles' section."
        )

    layout = layout_data["tiles"]

    log(f"Loaded layout: {config.STITCH_LAYOUT_PATH}")
    log(f"Dataset in layout: {layout_data.get('dataset')}")
    log(f"Tiles in layout: {len(layout)}")

    missing = [
        item["file"] for item in layout.values() if not Path(item["file"]).exists()
    ]

    if missing:
        log(f"Missing tile files: {len(missing)}")
        log(f"Example missing file: {missing[0]}")
        raise FileNotFoundError(
            "Some tile image paths inside stitch_layout.json do not exist."
        )

    full_width, full_height = stitcher.layout_canvas_size(layout)

    expected_levels = stitcher.compute_pyramid_levels(
        full_width,
        full_height,
        min_size=config.PYRAMID_MIN_SIZE,
    )

    log(f"Canvas:           {full_width} x {full_height}")
    log(f"Pyramid min size: {config.PYRAMID_MIN_SIZE}")
    log(f"Expected levels:  {len(expected_levels)}")

    for level in expected_levels:
        log(
            f"  level {level['level']}: "
            f"{level['width']} x {level['height']} "
            f"scale={level['scale']:.8f}"
        )

    omitted_tile_ids = (
        set(layout_data.get("all_failed_tile_ids", []))
        if config.OMIT_ALL_FAILED_TILES_FROM_STITCH
        else set()
    )

    if omitted_tile_ids:
        log(f"Omitting all-failed tiles: {len(omitted_tile_ids)}")
    else:
        log("Omitting all-failed tiles: 0")

    log("\nWriting pyramidal OME-TIFF...")
    pyramid_info = stitcher.write_pyramidal_ome_tiff(
        layout,
        config.PYRAMID_OME_TIFF_PATH,
        omitted_tile_ids=omitted_tile_ids,
    )

    # Keep the globally refined layout, but update the pyramid metadata
    # to exactly reflect what stage 7 wrote.
    layout_data["pyramid_info"] = pyramid_info
    layout_data["pyramid_info"]["pyramid_min_size"] = int(config.PYRAMID_MIN_SIZE)
    layout_data["pyramid_info"]["tile_size"] = int(config.PYRAMID_TILE_SIZE)
    layout_data["pyramid_info"]["compression"] = config.PYRAMID_COMPRESSION
    layout_data["tiles"] = layout
    layout_data["export_note"] = (
        "OME-TIFF exported from globally refined stitch_layout.json."
    )

    save_json(layout_data, config.STITCH_LAYOUT_PATH)

    section("STITCH SUMMARY")
    log(f"OME-TIFF: {config.PYRAMID_OME_TIFF_PATH}")
    log(f"Layout:   {config.STITCH_LAYOUT_PATH}")
    log(
        f"Canvas:   {pyramid_info['full_width']} x "
        f"{pyramid_info['full_height']}  |  "
        f"{len(pyramid_info['levels'])} pyramid level(s)  |  "
        f"blend={config.BLEND_MODE}  |  "
        f"compression={config.PYRAMID_COMPRESSION}"
    )


if __name__ == "__main__":
    main()
