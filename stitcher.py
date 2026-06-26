"""
stitcher.py
===========
Stitching machinery (ported from "Gustaf's stitching" cell):

  - build_stitch_layout()       : metadata-scaled BFS placement of good edges,
                                  with disconnected components aligned to the
                                  filename-coordinate grid.
  - neighbor_average_fallback() : place all-failed tiles from their neighbors'
                                  positions (without trusting bad transforms).
  - write_stitched_image()      : flat PNG (downscaled if huge).
  - write_pyramidal_ome_tiff()  : memory-safe pyramidal OME-TIFF, rendered one
                                  level at a time into disk-backed memmaps.

All placement uses the strip-shift -> tile-offset conversion
(relative_offset_b_from_a) so it is geometrically consistent with the matcher.
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from collections import OrderedDict, defaultdict, deque

import config
import common
from common import log, edge_crop_origin, read_gray, read_color

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Tile shapes + strip-shift -> tile-offset
# ---------------------------------------------------------------------------
def get_tile_shapes(tiles):
    shapes = {}
    for tile in tiles:
        img = read_gray(tile["file"])
        if img is None:
            raise RuntimeError(f"Could not read image: {tile['file']}")
        shapes[tile["id"]] = img.shape[:2]
    return shapes


def relative_offset_b_from_a(result, tile_shapes):
    """top_left_b - top_left_a = origin_a - origin_b - shift."""
    if "dx" not in result or "dy" not in result:
        return None
    a_id = result["a"]["slide"]
    b_id = result["b"]["slide"]
    origin_a = edge_crop_origin(tile_shapes[a_id], result["a"]["edge"],
                                result.get("edge_strip", config.EDGE_STRIP))
    origin_b = edge_crop_origin(tile_shapes[b_id], result["b"]["edge"],
                                result.get("edge_strip", config.EDGE_STRIP))
    shift = np.array([result["dx"], result["dy"]], dtype=np.float64)
    return origin_a - origin_b - shift


def estimate_metadata_scale(results, tile_shapes):
    xs, ys = [], []
    for r in results.values():
        if r.get("status") not in config.STITCH_USABLE_STATUSES:
            continue
        rel = relative_offset_b_from_a(r, tile_shapes)
        if rel is None:
            continue
        dc = (np.array(r["b"]["coord"], float) - np.array(r["a"]["coord"], float))
        if abs(dc[0]) > 1e-9:
            s = rel[0] / dc[0]
            if np.isfinite(s) and s > 0:
                xs.append(s)
        if abs(dc[1]) > 1e-9:
            s = rel[1] / dc[1]
            if np.isfinite(s) and s > 0:
                ys.append(s)
    sx = float(np.median(xs)) if xs else None
    sy = float(np.median(ys)) if ys else None
    if sx is None and sy is None:
        sx = sy = 1.0
    elif sx is None:
        sx = sy
    elif sy is None:
        sy = sx
    return sx, sy


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
def build_stitch_layout(tiles, results, tile_shapes):
    tile_by_id = {t["id"]: t for t in tiles}
    scale_x, scale_y = estimate_metadata_scale(results, tile_shapes)

    min_sx = min(t["x"] for t in tiles)
    min_sy = min(t["y"] for t in tiles)
    meta_pos = {t["id"]: np.array([(t["x"] - min_sx) * scale_x,
                                   (t["y"] - min_sy) * scale_y]) for t in tiles}

    adjacency = defaultdict(list)
    for r in results.values():
        if r.get("status") not in config.STITCH_USABLE_STATUSES:
            continue
        rel = relative_offset_b_from_a(r, tile_shapes)
        if rel is None:
            continue
        a_id, b_id = r["a"]["slide"], r["b"]["slide"]
        adjacency[a_id].append((b_id, rel))
        adjacency[b_id].append((a_id, -rel))

    order = sorted(tile_by_id, key=lambda t: (meta_pos[t][1], meta_pos[t][0], t))
    unvisited = set(tile_by_id)
    components = []
    for start in order:
        if start not in unvisited:
            continue
        local = {start: np.array([0.0, 0.0])}
        q = deque([start]); unvisited.remove(start)
        while q:
            cur = q.popleft()
            for nb, rel in adjacency[cur]:
                if nb not in local:
                    local[nb] = local[cur] + rel
                    q.append(nb); unvisited.discard(nb)
        components.append(local)

    final = {}
    for comp in components:
        offs = np.array([meta_pos[t] - p for t, p in comp.items()])
        trans = np.median(offs, axis=0)
        for t, p in comp.items():
            final[t] = p + trans

    min_x = min(p[0] for p in final.values())
    min_y = min(p[1] for p in final.values())
    layout = OrderedDict()
    for t in order:
        x = int(round(final[t][0] - min_x))
        y = int(round(final[t][1] - min_y))
        h, w = tile_shapes[t]
        layout[str(t)] = {"x": x, "y": y, "width": int(w), "height": int(h),
                          "file": tile_by_id[t]["file"],
                          "stage_coord": [tile_by_id[t]["x"], tile_by_id[t]["y"]]}
    return layout, scale_x, scale_y


def _normalize_layout(layout):
    if not layout:
        return
    min_x = min(it["x"] for it in layout.values())
    min_y = min(it["y"] for it in layout.values())
    if min_x == 0 and min_y == 0:
        return
    for it in layout.values():
        it["x"] -= min_x
        it["y"] -= min_y


def neighbor_average_fallback(tiles, graph, layout, all_failed, no_conn,
                              scale_x, scale_y):
    """Place all-failed tiles from already-placed neighbors via stage coords."""
    if not config.PLACE_ALL_FAILED_TILES_FROM_NEIGHBOR_AVERAGE:
        return {"enabled": False}

    tbid = {t["id"]: t for t in tiles}
    all_failed = set(all_failed or set())
    no_conn = set(no_conn or set())

    neighbors = defaultdict(set)
    for conn in graph.values():
        a, b = int(conn["a"]["slide"]), int(conn["b"]["slide"])
        neighbors[a].add(b); neighbors[b].add(a)

    anchors = {t["id"] for t in tiles
               if t["id"] not in all_failed and str(t["id"]) in layout}
    pending = {t for t in all_failed if t not in no_conn and str(t) in layout}

    placed = OrderedDict()
    for _ in range(config.NEIGHBOR_FALLBACK_MAX_ITERS):
        changed = False
        for tid in sorted(pending):
            tile = tbid.get(tid)
            if tile is None:
                continue
            preds = []
            for nb in sorted(neighbors.get(tid, [])):
                if nb in anchors and str(nb) in layout and nb in tbid:
                    nt, ni = tbid[nb], layout[str(nb)]
                    preds.append(np.array([ni["x"] + (tile["x"] - nt["x"]) * scale_x,
                                           ni["y"] + (tile["y"] - nt["y"]) * scale_y]))
            if not preds:
                continue
            avg = np.mean(np.stack(preds), axis=0)
            layout[str(tid)]["x"] = int(round(avg[0]))
            layout[str(tid)]["y"] = int(round(avg[1]))
            placed[str(tid)] = {"neighbor_predictions_used": len(preds),
                                "x": layout[str(tid)]["x"], "y": layout[str(tid)]["y"]}
            pending.discard(tid); anchors.add(tid); changed = True
        if not changed:
            break
    _normalize_layout(layout)
    return {"enabled": True, "placed_by_neighbor_average": placed,
            "left_at_metadata_position": sorted(int(t) for t in pending | no_conn)}


# ---------------------------------------------------------------------------
# Status summary
# ---------------------------------------------------------------------------
def summarize_statuses(tiles, graph, results):
    summ = {t["id"]: {"total": 0, "good": 0, "suspicious": 0, "failed": 0}
            for t in tiles}
    for name, conn in graph.items():
        st = results.get(name, {}).get("status", "failed")
        if st not in ("good", "suspicious", "failed"):
            st = "failed"
        for side in ("a", "b"):
            tid = conn[side]["slide"]
            if tid in summ:
                summ[tid]["total"] += 1
                summ[tid][st] += 1
    all_failed = {tid for tid, s in summ.items()
                  if s["total"] > 0 and s["failed"] == s["total"]}
    no_conn = {tid for tid, s in summ.items() if s["total"] == 0}
    return summ, all_failed, no_conn


# ---------------------------------------------------------------------------
# Canvas helpers
# ---------------------------------------------------------------------------
def layout_canvas_size(layout):
    mx = max(it["x"] + it["width"] for it in layout.values())
    my = max(it["y"] + it["height"] for it in layout.values())
    return mx, my


def _output_scale(width, height):
    if config.MAX_STITCH_PIXELS is None:
        return 1.0
    px = width * height
    return 1.0 if px <= config.MAX_STITCH_PIXELS else math.sqrt(
        config.MAX_STITCH_PIXELS / px)


# ---------------------------------------------------------------------------
# Flat PNG
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Blending: per-tile weight maps
# ---------------------------------------------------------------------------
def _tile_weight(h, w, mode, feather):
    """Return an (h, w) float32 weight map for a tile.

    - "none"/"average": uniform weight 1 everywhere.
    - "feather": weight ramps linearly from ~0 at the edges to 1 in the
      interior over `feather` px, so tiles fade out toward their borders and
      overlaps blend smoothly. Distance-to-edge based, separable in x and y.
    """
    if mode != "feather" or feather <= 0:
        return np.ones((h, w), np.float32)

    fw = int(min(feather, max(1, w // 2), max(1, h // 2)))
    # ramp 0..1 over fw px at each side, flat 1 in the middle
    rx = np.ones(w, np.float32)
    ry = np.ones(h, np.float32)
    if fw > 0:
        ramp = (np.arange(1, fw + 1, dtype=np.float32)) / (fw + 1)
        rx[:fw] = ramp
        rx[-fw:] = ramp[::-1]
        ry[:fw] = ramp
        ry[-fw:] = ramp[::-1]
    wmap = np.outer(ry, rx)            # separable 2D weight
    # keep a small floor so a fully-feathered lone pixel still shows
    return np.maximum(wmap, 1e-3).astype(np.float32)


def write_stitched_image(layout, output_path, omitted_tile_ids=None):
    full_w, full_h = layout_canvas_size(layout)
    scale = _output_scale(full_w, full_h)
    ow = max(1, int(math.ceil(full_w * scale)))
    oh = max(1, int(math.ceil(full_h * scale)))
    if scale < 1.0:
        log(f"  large canvas ({full_w}x{full_h}) -> {scale:.3f}x ({ow}x{oh})")

    mode = config.BLEND_MODE
    omit = {str(t) for t in (omitted_tile_ids or set())}
    n_omit = 0

    if mode == "none":
        # fast path: last tile wins, no accumulation buffers
        canvas = np.full((oh, ow, 3), 255, np.uint8)
        for tid, it in layout.items():
            if str(tid) in omit:
                n_omit += 1
                continue
            img = read_color(it["file"])
            if img is None:
                log(f"  warn: cannot read tile {tid}: {it['file']}")
                continue
            if scale != 1.0:
                img = cv2.resize(img, (max(1, int(round(img.shape[1]*scale))),
                                       max(1, int(round(img.shape[0]*scale)))),
                                 interpolation=cv2.INTER_AREA)
            x = int(round(it["x"]*scale)); y = int(round(it["y"]*scale))
            h, w = img.shape[:2]
            x2, y2 = min(ow, x+w), min(oh, y+h)
            if x >= ow or y >= oh or x2 <= x or y2 <= y:
                continue
            canvas[y:y2, x:x2] = img[:y2-y, :x2-x]
    else:
        # blended path: accumulate color*weight and weight, then divide
        acc = np.zeros((oh, ow, 3), np.float32)
        wsum = np.zeros((oh, ow, 1), np.float32)
        feather = int(round(config.FEATHER_WIDTH * scale)) if scale != 1.0 \
            else config.FEATHER_WIDTH
        for tid, it in layout.items():
            if str(tid) in omit:
                n_omit += 1
                continue
            img = read_color(it["file"])
            if img is None:
                log(f"  warn: cannot read tile {tid}: {it['file']}")
                continue
            if scale != 1.0:
                img = cv2.resize(img, (max(1, int(round(img.shape[1]*scale))),
                                       max(1, int(round(img.shape[0]*scale)))),
                                 interpolation=cv2.INTER_AREA)
            x = int(round(it["x"]*scale)); y = int(round(it["y"]*scale))
            h, w = img.shape[:2]
            x2, y2 = min(ow, x+w), min(oh, y+h)
            if x >= ow or y >= oh or x2 <= x or y2 <= y:
                continue
            iw, ih = x2 - x, y2 - y
            wmap = _tile_weight(h, w, mode, feather)[:ih, :iw, None]
            acc[y:y2, x:x2] += img[:ih, :iw].astype(np.float32) * wmap
            wsum[y:y2, x:x2] += wmap
        covered = wsum[..., 0] > 0
        canvas = np.full((oh, ow, 3), 255, np.uint8)
        blended = (acc / np.maximum(wsum, 1e-6))
        canvas[covered] = np.clip(blended[covered], 0, 255).astype(np.uint8)

    if not cv2.imwrite(str(output_path), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise RuntimeError(f"Could not write stitched image: {output_path}")
    return {"full_canvas_width": int(full_w), "full_canvas_height": int(full_h),
            "output_width": ow, "output_height": oh, "output_scale": float(scale),
            "blend_mode": mode, "omitted_all_failed_tiles": n_omit}


# ---------------------------------------------------------------------------
# Pyramidal OME-TIFF (memory-safe)
# ---------------------------------------------------------------------------
def compute_pyramid_levels(full_w, full_h, min_size=1024):
    levels = []
    scale = 1.0
    while True:
        lw = max(1, int(math.ceil(full_w * scale)))
        lh = max(1, int(math.ceil(full_h * scale)))
        levels.append({"level": len(levels), "scale": scale,
                       "width": lw, "height": lh})
        if max(lw, lh) <= min_size:
            break
        scale *= 0.5
    return levels


def _render_level_to_memmap(layout, level_info, temp_dir, omit, bg=255):
    """Render one pyramid level into a disk-backed uint8 memmap (RGB).

    Blending (config.BLEND_MODE) is done with disk-backed float32 accumulators
    so even gigapixel levels stay off the RAM heap: we keep acc = sum(color*w)
    and wsum = sum(w) as memmaps, then divide into the final uint8 memmap.
    """
    level, scale = level_info["level"], level_info["scale"]
    ow, oh = level_info["width"], level_info["height"]
    mode = config.BLEND_MODE
    omit = {str(t) for t in (omit or set())}

    out_path = os.path.join(temp_dir, f"level_{level}.dat")
    out = np.memmap(out_path, dtype=np.uint8, mode="w+", shape=(oh, ow, 3))
    out[:] = bg

    if mode == "none":
        for tid, it in layout.items():
            if str(tid) in omit:
                continue
            img_bgr = read_color(it["file"])
            if img_bgr is None:
                log(f"  warn: cannot read tile {tid}: {it['file']}")
                continue
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            if scale != 1.0:
                img = cv2.resize(img, (max(1, int(round(img.shape[1]*scale))),
                                       max(1, int(round(img.shape[0]*scale)))),
                                 interpolation=cv2.INTER_AREA)
            x = int(round(it["x"]*scale)); y = int(round(it["y"]*scale))
            h, w = img.shape[:2]
            x2, y2 = min(ow, x+w), min(oh, y+h)
            if x >= ow or y >= oh or x2 <= x or y2 <= y:
                continue
            out[y:y2, x:x2] = img[:y2-y, :x2-x]
        out.flush()
        return np.memmap(out_path, dtype=np.uint8, mode="r",
                         shape=(oh, ow, 3)), out_path

    # blended path: disk-backed float accumulators
    acc_path = os.path.join(temp_dir, f"level_{level}_acc.dat")
    w_path = os.path.join(temp_dir, f"level_{level}_w.dat")
    acc = np.memmap(acc_path, dtype=np.float32, mode="w+", shape=(oh, ow, 3))
    wsum = np.memmap(w_path, dtype=np.float32, mode="w+", shape=(oh, ow, 1))
    acc[:] = 0.0
    wsum[:] = 0.0
    feather = int(round(config.FEATHER_WIDTH * scale)) if scale != 1.0 \
        else config.FEATHER_WIDTH

    for tid, it in layout.items():
        if str(tid) in omit:
            continue
        img_bgr = read_color(it["file"])
        if img_bgr is None:
            log(f"  warn: cannot read tile {tid}: {it['file']}")
            continue
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if scale != 1.0:
            img = cv2.resize(img, (max(1, int(round(img.shape[1]*scale))),
                                   max(1, int(round(img.shape[0]*scale)))),
                             interpolation=cv2.INTER_AREA)
        x = int(round(it["x"]*scale)); y = int(round(it["y"]*scale))
        h, w = img.shape[:2]
        x2, y2 = min(ow, x+w), min(oh, y+h)
        if x >= ow or y >= oh or x2 <= x or y2 <= y:
            continue
        iw, ih = x2 - x, y2 - y
        wmap = _tile_weight(h, w, mode, feather)[:ih, :iw, None]
        acc[y:y2, x:x2] += img[:ih, :iw].astype(np.float32) * wmap
        wsum[y:y2, x:x2] += wmap

    # divide in row blocks to keep peak RAM low
    block = max(1, min(oh, 2048))
    for r0 in range(0, oh, block):
        r1 = min(oh, r0 + block)
        w_blk = np.asarray(wsum[r0:r1])
        covered = w_blk[..., 0] > 0
        if covered.any():
            blended = np.asarray(acc[r0:r1]) / np.maximum(w_blk, 1e-6)
            out_blk = np.asarray(out[r0:r1])
            out_blk[covered] = np.clip(blended[covered], 0, 255).astype(np.uint8)
            out[r0:r1] = out_blk
    out.flush()

    # clean up the accumulator memmaps for this level
    del acc, wsum
    for p in (acc_path, w_path):
        try:
            os.remove(p)
        except OSError:
            pass
    return np.memmap(out_path, dtype=np.uint8, mode="r",
                     shape=(oh, ow, 3)), out_path


def write_pyramidal_ome_tiff(layout, output_path, omitted_tile_ids=None,
                             tile_size=None, min_pyramid_size=None,
                             compression=None):
    common.ensure_requirements({"tifffile": "tifffile"})
    import tifffile

    tile_size = tile_size or config.PYRAMID_TILE_SIZE
    min_pyramid_size = min_pyramid_size or config.PYRAMID_MIN_SIZE
    compression = compression or config.PYRAMID_COMPRESSION

    full_w, full_h = layout_canvas_size(layout)
    levels = compute_pyramid_levels(full_w, full_h, min_pyramid_size)
    log("  pyramid levels:")
    for info in levels:
        log(f"    level {info['level']}: {info['width']}x{info['height']} "
            f"scale={info['scale']:.6f}")

    temp_dir = tempfile.mkdtemp(prefix="pyramid_render_")
    arrays = []
    try:
        for info in levels:
            log(f"  rendering level {info['level']}...")
            arr, _ = _render_level_to_memmap(layout, info, temp_dir,
                                             omitted_tile_ids)
            arrays.append(arr)

        opts = {"photometric": "rgb", "tile": (tile_size, tile_size),
                "compression": compression, "metadata": {"axes": "YXS"}}
        log(f"  writing OME-TIFF: {output_path}")
        with tifffile.TiffWriter(str(output_path), bigtiff=True, ome=True) as tif:
            tif.write(arrays[0], subifds=len(arrays) - 1, **opts)
            for arr in arrays[1:]:
                tif.write(arr, subfiletype=1, **opts)
        return {"output_path": str(output_path),
                "full_width": int(full_w), "full_height": int(full_h),
                "levels": levels, "tile_size": int(tile_size),
                "compression": compression}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
