#!/usr/bin/env python3
"""
06_global_layout.py
===================
Build globally refined stitch_layout.json from stitch_graph.json and
transforms_translation.json.

This stage does NOT write the OME-TIFF. It only creates:

    output/stitch_layout.json

The OME-TIFF writer in stage 07 reads this JSON and uses its tile positions.
"""

from __future__ import annotations

import heapq
import math
import sys
from collections import defaultdict, deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

import config
import common
import stitcher

from common import (
    log,
    section,
    save_json,
    load_json,
    load_tiles,
    load_tiles_from_graph,
    edge_crop_origin,
)


USED_STATUSES_FOR_LAYOUT = config.STITCH_USABLE_STATUSES


def load_tile_records(tile_list):
    tile_records = {}

    for tile in tile_list:
        img = cv2.imread(tile["file"], cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise FileNotFoundError(f"Could not load tile image: {tile['file']}")

        height, width = img.shape[:2]

        tile_records[int(tile["id"])] = {
            "file": tile["file"],
            "stage_coord": [int(tile["x"]), int(tile["y"])],
            "width": int(width),
            "height": int(height),
        }

    return tile_records


def load_layout_edges(graph, results, tile_records, used_statuses):
    """
    Convert strip-local dx/dy into full-tile relative offsets.

    transforms_translation.json stores:

        local_shift = point_b_in_strip - point_a_in_strip

    But layout optimization needs:

        tile_position_b - tile_position_a

    Conversion:

        tile_shift = origin_a - origin_b - local_shift
    """
    layout_edges = {}
    incident_statuses = defaultdict(list)

    for name, conn in graph.items():
        a_id = int(conn["a"]["slide"])
        b_id = int(conn["b"]["slide"])

        result = results.get(name, {})
        status = result.get("status", "failed")

        incident_statuses[a_id].append(status)
        incident_statuses[b_id].append(status)

        if status not in used_statuses:
            continue

        if "dx" not in result or "dy" not in result:
            continue

        local_dx = float(result["dx"])
        local_dy = float(result["dy"])

        strip = int(result.get("edge_strip", config.EDGE_STRIP))

        shape_a = (
            tile_records[a_id]["height"],
            tile_records[a_id]["width"],
        )

        shape_b = (
            tile_records[b_id]["height"],
            tile_records[b_id]["width"],
        )

        origin_a = edge_crop_origin(
            shape_a,
            conn["a"]["edge"],
            strip,
        )

        origin_b = edge_crop_origin(
            shape_b,
            conn["b"]["edge"],
            strip,
        )

        local_shift = np.array([local_dx, local_dy], dtype=float)

        tile_shift = origin_a - origin_b - local_shift

        inliers = float(result.get("inliers", 0))
        inlier_ratio = float(result.get("inlier_ratio", 0.0))

        weight = max(inliers * inlier_ratio, 1e-9)

        layout_edges[name] = {
            "a_id": a_id,
            "b_id": b_id,
            "dx": float(tile_shift[0]),
            "dy": float(tile_shift[1]),
            "weight": float(weight),
            "status": status,
        }

    all_tile_ids = set(tile_records)

    no_connection_tile_ids = sorted(
        tile_id for tile_id in all_tile_ids if tile_id not in incident_statuses
    )

    all_failed_tile_ids = sorted(
        tile_id
        for tile_id in all_tile_ids
        if tile_id in incident_statuses
        and not any(status in used_statuses for status in incident_statuses[tile_id])
    )

    return layout_edges, all_failed_tile_ids, no_connection_tile_ids


def weighted_average(values, weights):
    if not values:
        return None

    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    return float(np.average(values, weights=np.maximum(weights, 1e-9)))


def estimate_metadata_scales(layout_edges, tile_records):
    """
    Estimate pixels per stage-coordinate unit from accepted edges.
    """
    scale_x_values = []
    scale_x_weights = []

    scale_y_values = []
    scale_y_weights = []

    for edge in layout_edges.values():
        record_a = tile_records[edge["a_id"]]
        record_b = tile_records[edge["b_id"]]

        stage_dx = record_b["stage_coord"][0] - record_a["stage_coord"][0]
        stage_dy = record_b["stage_coord"][1] - record_a["stage_coord"][1]

        if stage_dx != 0:
            scale_x_values.append(abs(edge["dx"] / stage_dx))
            scale_x_weights.append(edge["weight"])

        if stage_dy != 0:
            scale_y_values.append(abs(edge["dy"] / stage_dy))
            scale_y_weights.append(edge["weight"])

    scale_x = weighted_average(scale_x_values, scale_x_weights)
    scale_y = weighted_average(scale_y_values, scale_y_weights)

    stage_xs = sorted({record["stage_coord"][0] for record in tile_records.values()})
    stage_ys = sorted({record["stage_coord"][1] for record in tile_records.values()})

    if scale_x is None:
        stage_x_gaps = np.diff(stage_xs)
        median_width = np.median([record["width"] for record in tile_records.values()])

        if len(stage_x_gaps) == 0:
            scale_x = 1.0
        else:
            scale_x = float(median_width / np.median(stage_x_gaps))

    if scale_y is None:
        stage_y_gaps = np.diff(stage_ys)
        median_height = np.median(
            [record["height"] for record in tile_records.values()]
        )

        if len(stage_y_gaps) == 0:
            scale_y = 1.0
        else:
            scale_y = float(median_height / np.median(stage_y_gaps))

    return float(scale_x), float(scale_y)


def build_metadata_positions(tile_records, scale_x, scale_y):
    min_stage_x = min(record["stage_coord"][0] for record in tile_records.values())
    min_stage_y = min(record["stage_coord"][1] for record in tile_records.values())

    metadata_positions = {}

    for tile_id, record in tile_records.items():
        stage_x, stage_y = record["stage_coord"]

        metadata_positions[tile_id] = np.array(
            [
                (stage_x - min_stage_x) * scale_x,
                (stage_y - min_stage_y) * scale_y,
            ],
            dtype=float,
        )

    return metadata_positions


def build_layout_adjacency(layout_edges):
    adjacency = defaultdict(list)

    for name, edge in layout_edges.items():
        a_id = edge["a_id"]
        b_id = edge["b_id"]
        dx = edge["dx"]
        dy = edge["dy"]
        weight = edge["weight"]

        adjacency[a_id].append((b_id, name, dx, dy, weight))
        adjacency[b_id].append((a_id, name, -dx, -dy, weight))

    return adjacency


def maximum_spanning_forest(active_tile_ids, adjacency):
    """
    Build one maximum spanning tree per connected component.
    """
    unvisited = set(active_tile_ids)

    tree_adjacency = defaultdict(list)
    tree_edge_names = set()
    component_roots = []

    while unvisited:
        root = min(unvisited)

        component_roots.append(root)
        unvisited.remove(root)

        component_visited = {root}
        frontier = []

        for neighbor_id, name, dx, dy, weight in adjacency[root]:
            heapq.heappush(
                frontier,
                (-weight, name, root, neighbor_id, dx, dy),
            )

        while frontier:
            negative_weight, name, from_id, neighbor_id, dx, dy = heapq.heappop(
                frontier
            )

            if neighbor_id in component_visited:
                continue

            weight = -negative_weight

            component_visited.add(neighbor_id)
            unvisited.discard(neighbor_id)
            tree_edge_names.add(name)

            tree_adjacency[from_id].append((neighbor_id, name, dx, dy, weight))
            tree_adjacency[neighbor_id].append((from_id, name, -dx, -dy, weight))

            for next_id, next_name, next_dx, next_dy, next_weight in adjacency[
                neighbor_id
            ]:
                if next_id not in component_visited:
                    heapq.heappush(
                        frontier,
                        (
                            -next_weight,
                            next_name,
                            neighbor_id,
                            next_id,
                            next_dx,
                            next_dy,
                        ),
                    )

    return tree_adjacency, component_roots, tree_edge_names


def propagate_forest_positions(
    active_tile_ids,
    tree_adjacency,
    component_roots,
    metadata_positions,
):
    """
    Initialize every connected component from metadata position, then propagate
    through its maximum spanning tree.
    """
    positions = {}
    seen = set()
    queue = deque()

    for root in component_roots:
        positions[root] = metadata_positions[root].copy()
        seen.add(root)
        queue.append(root)

    while queue:
        current = queue.popleft()

        for neighbor_id, name, dx, dy, weight in tree_adjacency[current]:
            if neighbor_id in seen:
                continue

            positions[neighbor_id] = positions[current] + np.array(
                [dx, dy], dtype=float
            )

            seen.add(neighbor_id)
            queue.append(neighbor_id)

    missing = set(active_tile_ids) - seen

    for tile_id in missing:
        positions[tile_id] = metadata_positions[tile_id].copy()

    return positions


def refine_with_least_squares(
    active_tile_ids,
    layout_edges,
    initial_positions,
    component_roots,
):
    """
    Solve globally optimized tile positions.

    For each accepted edge:

        position_b - position_a ~= measured_shift

    One root per component is pinned to the tree-propagated position.
    """
    if not active_tile_ids:
        return {}, {}

    tile_ids = sorted(active_tile_ids)

    index_of = {tile_id: index for index, tile_id in enumerate(tile_ids)}

    n = len(tile_ids)
    pin_weight = 1e6

    edge_list = []

    for name, edge in layout_edges.items():
        edge_list.append(
            {
                "name": name,
                "a_idx": index_of[edge["a_id"]],
                "b_idx": index_of[edge["b_id"]],
                "dx": edge["dx"],
                "dy": edge["dy"],
                "weight": edge["weight"],
            }
        )

    def solve_axis(axis):
        rows = []
        rhs = []

        for edge in edge_list:
            row = np.zeros(n, dtype=float)

            row[edge["b_idx"]] = 1.0
            row[edge["a_idx"]] = -1.0

            value = edge["dx"] if axis == "x" else edge["dy"]
            sqrt_weight = np.sqrt(max(edge["weight"], 1e-9))

            rows.append(row * sqrt_weight)
            rhs.append(value * sqrt_weight)

        for root in component_roots:
            pin_row = np.zeros(n, dtype=float)
            pin_row[index_of[root]] = 1.0

            root_value = (
                initial_positions[root][0]
                if axis == "x"
                else initial_positions[root][1]
            )

            rows.append(pin_row * np.sqrt(pin_weight))
            rhs.append(root_value * np.sqrt(pin_weight))

        A = np.vstack(rows)
        b = np.asarray(rhs, dtype=float)

        solution, *_ = np.linalg.lstsq(A, b, rcond=None)

        return solution

    x_solution = solve_axis("x")
    y_solution = solve_axis("y")

    refined_positions = {
        tile_id: np.array(
            [
                x_solution[index_of[tile_id]],
                y_solution[index_of[tile_id]],
            ],
            dtype=float,
        )
        for tile_id in tile_ids
    }

    residuals = {}

    for name, edge in layout_edges.items():
        predicted_shift = (
            refined_positions[edge["b_id"]] - refined_positions[edge["a_id"]]
        )

        residuals[name] = {
            "dx_residual": float(predicted_shift[0] - edge["dx"]),
            "dy_residual": float(predicted_shift[1] - edge["dy"]),
        }

    return refined_positions, residuals


def build_tile_neighbors(graph):
    neighbors = defaultdict(set)

    for conn in graph.values():
        a_id = int(conn["a"]["slide"])
        b_id = int(conn["b"]["slide"])

        neighbors[a_id].add(b_id)
        neighbors[b_id].add(a_id)

    return neighbors


def place_failed_tiles_by_neighbor_average(
    refined_positions,
    all_failed_tile_ids,
    no_connection_tile_ids,
    graph,
    tile_records,
    metadata_positions,
    scale_x,
    scale_y,
):
    """
    Place tiles without accepted connections using already-positioned neighbors
    and scaled stage-coordinate differences.
    """
    positions = {
        tile_id: position.copy() for tile_id, position in refined_positions.items()
    }

    neighbors = build_tile_neighbors(graph)

    placed_counts = {}
    unresolved = sorted(all_failed_tile_ids)

    while unresolved:
        next_unresolved = []
        progress = False

        for tile_id in unresolved:
            predictions = []

            stage_x, stage_y = tile_records[tile_id]["stage_coord"]

            for neighbor_id in sorted(neighbors[tile_id]):
                if neighbor_id not in positions:
                    continue

                neighbor_stage_x, neighbor_stage_y = tile_records[neighbor_id][
                    "stage_coord"
                ]

                metadata_delta = np.array(
                    [
                        (stage_x - neighbor_stage_x) * scale_x,
                        (stage_y - neighbor_stage_y) * scale_y,
                    ],
                    dtype=float,
                )

                predictions.append(positions[neighbor_id] + metadata_delta)

            if predictions:
                positions[tile_id] = np.mean(predictions, axis=0)
                placed_counts[tile_id] = len(predictions)
                progress = True
            else:
                next_unresolved.append(tile_id)

        if not progress:
            unresolved = next_unresolved
            break

        unresolved = next_unresolved

    left_at_metadata_position = []

    for tile_id in unresolved:
        positions[tile_id] = metadata_positions[tile_id].copy()
        left_at_metadata_position.append(tile_id)

    for tile_id in no_connection_tile_ids:
        positions[tile_id] = metadata_positions[tile_id].copy()
        left_at_metadata_position.append(tile_id)

    for tile_id in tile_records:
        if tile_id not in positions:
            positions[tile_id] = metadata_positions[tile_id].copy()

            if tile_id not in left_at_metadata_position:
                left_at_metadata_position.append(tile_id)

    return positions, placed_counts, sorted(left_at_metadata_position)


def normalize_and_round_positions(positions):
    min_x = min(position[0] for position in positions.values())
    min_y = min(position[1] for position in positions.values())

    normalized = {}

    for tile_id, position in positions.items():
        normalized[tile_id] = {
            "x": int(round(position[0] - min_x)),
            "y": int(round(position[1] - min_y)),
        }

    return normalized


def build_global_layout_json(tiles, graph, results):
    tile_records = load_tile_records(tiles)

    layout_edges, all_failed_tile_ids, no_connection_tile_ids = load_layout_edges(
        graph=graph,
        results=results,
        tile_records=tile_records,
        used_statuses=USED_STATUSES_FOR_LAYOUT,
    )

    metadata_scale_x, metadata_scale_y = estimate_metadata_scales(
        layout_edges,
        tile_records,
    )

    metadata_positions = build_metadata_positions(
        tile_records,
        metadata_scale_x,
        metadata_scale_y,
    )

    active_tile_ids = set()

    for edge in layout_edges.values():
        active_tile_ids.add(edge["a_id"])
        active_tile_ids.add(edge["b_id"])

    layout_adjacency = build_layout_adjacency(layout_edges)

    tree_adjacency, component_roots, tree_edge_names = maximum_spanning_forest(
        active_tile_ids,
        layout_adjacency,
    )

    initial_positions = propagate_forest_positions(
        active_tile_ids,
        tree_adjacency,
        component_roots,
        metadata_positions,
    )

    refined_positions, residuals = refine_with_least_squares(
        active_tile_ids=active_tile_ids,
        layout_edges=layout_edges,
        initial_positions=initial_positions,
        component_roots=component_roots,
    )

    all_positions, failed_neighbor_counts, left_at_metadata_position = (
        place_failed_tiles_by_neighbor_average(
            refined_positions=refined_positions,
            all_failed_tile_ids=all_failed_tile_ids,
            no_connection_tile_ids=no_connection_tile_ids,
            graph=graph,
            tile_records=tile_records,
            metadata_positions=metadata_positions,
            scale_x=metadata_scale_x,
            scale_y=metadata_scale_y,
        )
    )

    final_positions = normalize_and_round_positions(all_positions)

    full_canvas_width = max(
        final_positions[tile_id]["x"] + tile_records[tile_id]["width"]
        for tile_id in tile_records
    )

    full_canvas_height = max(
        final_positions[tile_id]["y"] + tile_records[tile_id]["height"]
        for tile_id in tile_records
    )

    full_canvas_pixels = full_canvas_width * full_canvas_height

    if (
        config.MAX_STITCH_PIXELS is None
        or full_canvas_pixels <= config.MAX_STITCH_PIXELS
    ):
        output_scale = 1.0
    else:
        output_scale = float(np.sqrt(config.MAX_STITCH_PIXELS / full_canvas_pixels))

    output_width = int(math.ceil(full_canvas_width * output_scale))
    output_height = int(math.ceil(full_canvas_height * output_scale))

    # IMPORTANT:
    # Use PYRAMID_MIN_SIZE here, not PYRAMID_TILE_SIZE, so the JSON metadata
    # matches the actual OME-TIFF writer's stopping rule.
    pyramid_levels = stitcher.compute_pyramid_levels(
        full_canvas_width,
        full_canvas_height,
        min_size=config.PYRAMID_MIN_SIZE,
    )

    placed_by_neighbor_average = {}

    for tile_id in sorted(failed_neighbor_counts):
        placed_by_neighbor_average[str(tile_id)] = {
            "neighbor_predictions_used": int(failed_neighbor_counts[tile_id]),
            "x": int(final_positions[tile_id]["x"]),
            "y": int(final_positions[tile_id]["y"]),
        }

    ordered_tile_ids = sorted(
        tile_records,
        key=lambda tile_id: (
            tile_records[tile_id]["stage_coord"][1],
            tile_records[tile_id]["stage_coord"][0],
        ),
    )

    tiles_output = {}

    for tile_id in ordered_tile_ids:
        record = tile_records[tile_id]
        position = final_positions[tile_id]

        tiles_output[str(tile_id)] = {
            "x": int(position["x"]),
            "y": int(position["y"]),
            "width": int(record["width"]),
            "height": int(record["height"]),
            "file": record["file"],
            "stage_coord": [
                int(record["stage_coord"][0]),
                int(record["stage_coord"][1]),
            ],
        }

    layout_json = {
        "dataset": config.DATASET_NAME,
        "source_graph": str(config.GRAPH_PATH),
        "source_transforms": str(config.TRANSFORMS_PATH),
        "match_method": config.MATCH_METHOD,
        "blend_mode": config.BLEND_MODE,
        "used_statuses_for_layout": sorted(USED_STATUSES_FOR_LAYOUT),
        "metadata_scale_x": float(metadata_scale_x),
        "metadata_scale_y": float(metadata_scale_y),
        "global_refinement": {
            "enabled": True,
            "accepted_layout_edges": int(len(layout_edges)),
            "tree_edges": int(len(tree_edge_names)),
            "component_roots": [int(root) for root in component_roots],
            "residuals": residuals,
        },
        "stitch_info": {
            "full_canvas_width": int(full_canvas_width),
            "full_canvas_height": int(full_canvas_height),
            "output_width": int(output_width),
            "output_height": int(output_height),
            "output_scale": float(output_scale),
            "omitted_all_failed_tiles": (
                len(all_failed_tile_ids)
                if config.OMIT_ALL_FAILED_TILES_FROM_STITCH
                else 0
            ),
        },
        "pyramid_info": {
            "output_path": str(config.PYRAMID_OME_TIFF_PATH),
            "full_width": int(full_canvas_width),
            "full_height": int(full_canvas_height),
            "levels": pyramid_levels,
            "tile_size": int(config.PYRAMID_TILE_SIZE),
            "compression": config.PYRAMID_COMPRESSION,
            "pyramid_min_size": int(config.PYRAMID_MIN_SIZE),
        },
        "all_failed_tile_ids": [int(tile_id) for tile_id in all_failed_tile_ids],
        "no_connection_tile_ids": [int(tile_id) for tile_id in no_connection_tile_ids],
        "all_failed_tiles_omitted_from_stitch": bool(
            config.OMIT_ALL_FAILED_TILES_FROM_STITCH
        ),
        "all_failed_tile_neighbor_average_fallback": {
            "enabled": True,
            "placed_by_neighbor_average": placed_by_neighbor_average,
            "left_at_metadata_position": [
                int(tile_id) for tile_id in left_at_metadata_position
            ],
        },
        "tiles": tiles_output,
    }

    return layout_json


def main() -> None:
    section("STAGE 6  -  GLOBAL LAYOUT REFINEMENT")
    config.validate_dataset()

    if not config.GRAPH_PATH.exists():
        raise FileNotFoundError(
            f"Missing {config.GRAPH_PATH}. Run 01_build_borders.py first."
        )

    if not config.TRANSFORMS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {config.TRANSFORMS_PATH}. Run 03_match.py first."
        )

    graph = load_json(config.GRAPH_PATH)
    results = load_json(config.TRANSFORMS_PATH)

    try:
        tiles = load_tiles(config.input_folder())
    except FileNotFoundError:
        tiles = []

    if not tiles:
        log("Input folder unavailable; reconstructing tiles from graph.")
        tiles = load_tiles_from_graph(graph)

    log(f"Dataset:      {config.DATASET_NAME}")
    log(f"Input folder: {config.input_folder()}")
    log(f"Output dir:   {config.OUTPUT_DIR}")
    log(f"Tiles:        {len(tiles)}")
    log(f"Connections:  {len(graph)}")

    layout_json = build_global_layout_json(
        tiles=tiles,
        graph=graph,
        results=results,
    )

    save_json(layout_json, config.STITCH_LAYOUT_PATH)

    section("GLOBAL LAYOUT SUMMARY")
    log(f"Layout:          {config.STITCH_LAYOUT_PATH}")
    log(
        f"Canvas:          "
        f"{layout_json['pyramid_info']['full_width']} x "
        f"{layout_json['pyramid_info']['full_height']}"
    )
    log(f"Pyramid min size: {config.PYRAMID_MIN_SIZE}")
    log(f"Pyramid levels:   {len(layout_json['pyramid_info']['levels'])}")
    log(
        f"Accepted edges:   {layout_json['global_refinement']['accepted_layout_edges']}"
    )
    log(f"Tree edges:       {layout_json['global_refinement']['tree_edges']}")
    log(f"Component roots:  {layout_json['global_refinement']['component_roots']}")
    log(f"All-failed tiles: {len(layout_json['all_failed_tile_ids'])}")
    log(f"No-connection:    {len(layout_json['no_connection_tile_ids'])}")


if __name__ == "__main__":
    main()
