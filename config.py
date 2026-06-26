"""
config.py
=========
Central configuration for the slide-stitching pipeline.

Every script imports from here so paths, dataset selection, and parameters
live in ONE place. Edit this file (or override via environment variables /
CLI flags) instead of changing the scripts.

The output filenames intentionally match the names produced by the original
Colab notebooks, so downstream tools and notebooks keep working:
    stitch_graph.json
    transforms_translation.json
    metrics.json / metrics.csv
    global_positions_<metric>.json
    global_positions_optimized_<metric>.json
    stitched_image.png
"""

from __future__ import annotations

import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Dataset selection
# ---------------------------------------------------------------------------
# Choose one of the CellaVision pathology datasets.
VALID_DATASETS = {
    "cytology-small",
    "cytology-large",
    "histology-small",
    "histology-large",
}

# Override with the DATASET_NAME environment variable if you like:
#   DATASET_NAME=histology-large python scripts/01_build_borders.py
DATASET_NAME = os.environ.get("DATASET_NAME", "cytology-small")


# ---------------------------------------------------------------------------
# Filesystem paths (LOCAL)
# ---------------------------------------------------------------------------
# Root that contains the dataset folders. Override with INPUT_ROOT env var.
# Expected layout:
#   <INPUT_ROOT>/<DATASET_NAME>/Images-FOV/<id>_<x>_<y>.bmp
INPUT_ROOT = Path(
    os.environ.get(
        "INPUT_ROOT",
        # default: a sibling "data" folder next to the pipeline
        str(Path(__file__).resolve().parent / "data"),
    )
)

# Where all pipeline artifacts get written. Override with OUTPUT_DIR env var.
OUTPUT_DIR = Path(
    os.environ.get(
        "OUTPUT_DIR",
        str(Path(__file__).resolve().parent / "output"),
    )
)


def input_folder(dataset: str = DATASET_NAME) -> Path:
    """Folder holding the raw FOV tiles for a dataset."""
    return INPUT_ROOT / dataset / "Images-FOV"


# ---------------------------------------------------------------------------
# Canonical output file paths (names match the notebooks)
# ---------------------------------------------------------------------------
GRAPH_PATH = OUTPUT_DIR / "stitch_graph.json"
TRANSFORMS_PATH = OUTPUT_DIR / "transforms_translation.json"
METRICS_JSON = OUTPUT_DIR / "metrics.json"
METRICS_CSV = OUTPUT_DIR / "metrics.csv"

DEBUG_DIR = OUTPUT_DIR / "debug_matches"
STRIPS_DIR = OUTPUT_DIR / "border_strips"   # cut edge strips (stage 02)

STITCH_LAYOUT_PATH = OUTPUT_DIR / "stitch_layout.json"
STITCHED_IMAGE_PATH = OUTPUT_DIR / "stitched_image.png"
CONNECTION_OVERVIEW_PATH = OUTPUT_DIR / "connection_overview_numbered.png"
# (PYRAMID_OME_TIFF_PATH is defined in the pyramid section below)


def global_positions_path(metric: str) -> Path:
    return OUTPUT_DIR / f"global_positions_{metric}.json"


def global_positions_optimized_path(metric: str) -> Path:
    return OUTPUT_DIR / f"global_positions_optimized_{metric}.json"


# ---------------------------------------------------------------------------
# Algorithm parameters
# ---------------------------------------------------------------------------
# Tile filename pattern: <id>_<x>_<y>.bmp  (x, y may be negative)
TILE_REGEX = r"^(\d+)_(-?\d+)_(-?\d+)\.bmp$"

# Width (in pixels) of the border strip cut from each tile edge for matching.
EDGE_STRIP = 150

# SIFT / RANSAC matching parameters
RATIO_TEST = 0.75          # Lowe's ratio test threshold
RANSAC_THRESHOLD = 3.0     # inlier distance threshold (pixels)
MAX_MATCHES_TO_DRAW = 100  # cap on matches drawn in debug images

# Connection classification thresholds (mirror the notebook)
GOOD_MIN_INLIERS = 15
GOOD_MIN_INLIER_RATIO = 0.5


# ---------------------------------------------------------------------------
# Feature-matching method (stage 3)
# ---------------------------------------------------------------------------
# Which method computes the per-border shift:
#   "sift"   - classic SIFT keypoints + ratio test + translation RANSAC (CPU, fast)
#   "roma"   - RoMaV2 dense neural matcher + translation RANSAC (needs GPU + romav2)
#   "hybrid" - try SIFT first; if its match isn't confidently good, fall back to RoMa
# Override with MATCH_METHOD env var:  MATCH_METHOD=roma python scripts/03_match.py
MATCH_METHOD = os.environ.get("MATCH_METHOD", "sift")
VALID_MATCH_METHODS = {"sift", "roma", "hybrid"}

# RoMa samples the whole strip uniformly, so many samples land off-overlap and
# inlier_ratio is much lower than SIFT. These thresholds mirror the notebook.
ROMA_SAMPLES = 2000           # points sampled from the dense warp
ROMA_GOOD_MIN_INLIERS = 15
ROMA_GOOD_MIN_INLIER_RATIO = 0.1

# Hybrid: accept SIFT only when it is *confidently* good; otherwise fall back to
# RoMa. These can be stricter than the plain-SIFT "good" thresholds, since the
# whole point is to keep only the most trustworthy SIFT matches and let RoMa
# handle the rest (low-texture / blank-background borders SIFT struggles with).
HYBRID_SIFT_MIN_INLIERS = 20        # SIFT needs at least this many inliers...
HYBRID_SIFT_MIN_INLIER_RATIO = 0.6  # ...and at least this inlier ratio, else -> RoMa


# ---------------------------------------------------------------------------
# Pyramidal OME-TIFF output (stage 7)
# ---------------------------------------------------------------------------
PYRAMID_OME_TIFF_PATH = OUTPUT_DIR / "stitched_pyramid.ome.tif"
PYRAMID_TILE_SIZE = 512       # internal tile size of the OME-TIFF
PYRAMID_MIN_SIZE = 1024       # stop building pyramid levels below this size
PYRAMID_COMPRESSION = "deflate"   # "deflate" (lossless) or "jpeg" (smaller, lossy)

# Stitch placement / canvas controls (mirror Gustaf's cell)
STITCH_USABLE_STATUSES = {"good"}     # statuses used for geometric placement
MAX_STITCH_PIXELS = 250_000_000       # safety cap for the flat PNG; None = full res
OMIT_ALL_FAILED_TILES_FROM_STITCH = False
PLACE_ALL_FAILED_TILES_FROM_NEIGHBOR_AVERAGE = True
NEIGHBOR_FALLBACK_MAX_ITERS = 8
WHITE_OUT_ALL_FAILED_TILES_IN_OVERVIEW = True
OVERVIEW_THUMB_SIZE = 220
OVERVIEW_PADDING = 24

# ---------------------------------------------------------------------------
# Overlap blending (stage 7)
# ---------------------------------------------------------------------------
# How overlapping tiles are combined where they cover the same pixels:
#   "none"    - last tile wins (hard overwrite; fast, but visible seams)
#   "average" - mean of all tiles covering a pixel (simple, removes seams,
#               can look slightly soft if alignment is off by a pixel)
#   "feather" - distance-weighted blend: each tile fades out toward its edges,
#               so seams disappear smoothly. Best quality. (default)
BLEND_MODE = os.environ.get("BLEND_MODE", "feather")
VALID_BLEND_MODES = {"none", "average", "feather"}

# Width (px) of the feather ramp at each tile edge. Larger = softer transition.
# Capped to half the tile size internally.
FEATHER_WIDTH = 64

# Opposite-edge lookup used when building the neighbor graph.
OPPOSITE_EDGE = {
    "left": "right",
    "right": "left",
    "top": "bottom",
    "bottom": "top",
}


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------
def validate_dataset(dataset: str = DATASET_NAME) -> None:
    if dataset not in VALID_DATASETS:
        raise ValueError(
            f"Unknown dataset: {dataset!r}. "
            f"Choose one of: {sorted(VALID_DATASETS)}"
        )
