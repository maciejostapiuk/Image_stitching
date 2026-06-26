#!/usr/bin/env python3
"""
02_cut_borders.py
=================
STAGE 2 of the stitching pipeline: CUT the border strips that will be matched.

What it does
------------
For every connection in stitch_graph.json, reads the two tiles and cuts the
EDGE_STRIP-wide band along the shared border from each:
    - tile A's edge (e.g. its "right" strip)
    - tile B's opposite edge (e.g. its "left" strip)

These two strips are exactly the regions the SIFT matcher (stage 3) compares,
so cutting them once here makes the matching stage simpler, inspectable, and
re-runnable without touching full images.

Why cut to disk?
----------------
- You can eyeball the strips to debug bad borders.
- The matching stage reads small strips instead of full BMPs.
- Keeps the "what gets compared" decision in one explicit place.

Output
------
    output/border_strips/<connection>__a_<edge>.png
    output/border_strips/<connection>__b_<edge>.png
    output/border_strips/strips_index.json   (maps connection -> strip files)

Run
---
    python scripts/02_cut_borders.py
    EDGE_STRIP is taken from config.py (default 150).
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import common
from common import log, section, save_json, load_json, read_gray, crop_edge

import cv2


def cut_strip_for_side(side: dict, strip_width: int):
    """Read a tile and cut the strip along the given side's edge.

    Returns (strip_image, status_dict). status_dict is None on success,
    otherwise carries an error reason.
    """
    img = read_gray(side["file"])
    if img is None:
        return None, {"error": "could not load image", "file": side["file"]}

    strip = crop_edge(img, side["edge"], strip_width)
    if strip.size == 0:
        return None, {"error": "empty strip", "file": side["file"]}
    return strip, None


def main() -> None:
    section("STAGE 2  -  CUT BORDER STRIPS")
    config.validate_dataset()

    strip_width = config.EDGE_STRIP
    log(f"Edge strip width: {strip_width} px")

    if not config.GRAPH_PATH.exists():
        raise FileNotFoundError(
            f"Missing {config.GRAPH_PATH}. Run 01_build_borders.py first."
        )

    graph = load_json(config.GRAPH_PATH)
    log(f"Connections in graph: {len(graph)}")

    config.STRIPS_DIR.mkdir(parents=True, exist_ok=True)

    index: "OrderedDict" = OrderedDict()
    n_ok = 0
    n_fail = 0

    for name, conn in graph.items():
        entry = {
            "label": conn.get("label", ""),
            "type": conn.get("type", ""),
            "a_slide": conn["a"]["slide"],
            "b_slide": conn["b"]["slide"],
            "a_edge": conn["a"]["edge"],
            "b_edge": conn["b"]["edge"],
        }

        strip_a, err_a = cut_strip_for_side(conn["a"], strip_width)
        strip_b, err_b = cut_strip_for_side(conn["b"], strip_width)

        if err_a or err_b:
            entry["status"] = "failed"
            entry["reason"] = (err_a or err_b)["error"]
            index[name] = entry
            n_fail += 1
            log(f"  {name:24s} FAILED: {entry['reason']}")
            continue

        a_path = config.STRIPS_DIR / f"{name}__a_{conn['a']['edge']}.png"
        b_path = config.STRIPS_DIR / f"{name}__b_{conn['b']['edge']}.png"
        cv2.imwrite(str(a_path), strip_a)
        cv2.imwrite(str(b_path), strip_b)

        entry["status"] = "ok"
        entry["a_strip"] = str(a_path)
        entry["b_strip"] = str(b_path)
        entry["a_strip_wh"] = [int(strip_a.shape[1]), int(strip_a.shape[0])]
        entry["b_strip_wh"] = [int(strip_b.shape[1]), int(strip_b.shape[0])]
        index[name] = entry
        n_ok += 1

    save_json(index, config.STRIPS_DIR / "strips_index.json")

    log(f"\nStrips cut: {n_ok} ok, {n_fail} failed (of {len(graph)})")
    log(f"Strips written to: {config.STRIPS_DIR}")
    log("\nStage 2 complete.")


if __name__ == "__main__":
    main()
