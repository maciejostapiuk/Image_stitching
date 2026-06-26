"""
common.py
=========
Shared helpers used across pipeline stages: dependency bootstrap, tile
loading, edge cropping, JSON IO, and small logging utilities.

Importing this module triggers an automatic dependency check (see
`ensure_requirements`) so every script can simply `import common` and be
sure cv2 / numpy / etc. are present.
"""

from __future__ import annotations

import importlib
import json
import re
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import config


# ---------------------------------------------------------------------------
# Dependency bootstrap
# ---------------------------------------------------------------------------
# Map: import name -> pip spec. Kept minimal for the borders+cutting stage;
# later stages append their own (skimage, torch, lpips, scipy) on demand.
_CORE_REQUIREMENTS = {
    "cv2": "opencv-python",
    "numpy": "numpy",
}


def ensure_requirements(extra: Optional[dict] = None) -> None:
    """Install any missing dependencies via pip, then continue.

    `extra` maps import-name -> pip-spec for stage-specific packages.
    Safe to call multiple times; only missing packages are installed.
    """
    requirements = dict(_CORE_REQUIREMENTS)
    if extra:
        requirements.update(extra)

    missing = []
    for import_name, pip_spec in requirements.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(pip_spec)

    if missing:
        print(f"[setup] Installing missing packages: {missing}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", *missing]
        )
        importlib.invalidate_caches()


# Bootstrap the core deps as soon as common is imported.
ensure_requirements()

import cv2  # noqa: E402  (import after bootstrap)
import numpy as np  # noqa: E402


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    print(msg, flush=True)


def section(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n{title}\n{bar}", flush=True)


# ---------------------------------------------------------------------------
# JSON IO
# ---------------------------------------------------------------------------
def save_json(data, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    log(f"  saved: {path}")


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f, object_pairs_hook=OrderedDict)


# ---------------------------------------------------------------------------
# Tile loading
# ---------------------------------------------------------------------------
_TILE_RE = re.compile(config.TILE_REGEX, re.IGNORECASE)


def load_tiles(folder: Path) -> list[dict]:
    """Scan a folder for tile images named <id>_<x>_<y>.bmp.

    Returns a list of dicts: {id, x, y, file}.
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Missing input folder: {folder}")

    tiles = []
    for file in sorted(folder.iterdir()):
        if not file.is_file():
            continue
        m = _TILE_RE.match(file.name)
        if not m:
            continue
        tiles.append(
            {
                "id": int(m.group(1)),
                "x": int(m.group(2)),
                "y": int(m.group(3)),
                "file": str(file),
            }
        )
    return tiles


def load_tiles_from_graph(graph: dict) -> list[dict]:
    """Reconstruct a tile list from a stitch_graph.json (fallback when the
    raw image folder is unavailable)."""
    by_id = {}
    for conn in graph.values():
        for side in ("a", "b"):
            item = conn[side]
            tid = int(item["slide"])
            coord = item.get("coord", [0, 0])
            by_id[tid] = {
                "id": tid,
                "x": int(coord[0]),
                "y": int(coord[1]),
                "file": item["file"],
            }
    return sorted(by_id.values(), key=lambda t: (t["y"], t["x"], t["id"]))


# ---------------------------------------------------------------------------
# Image reading + edge cropping
# ---------------------------------------------------------------------------
def read_gray(path: str):
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def read_color(path: str):
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def crop_edge(img, edge: str, strip: int):
    """Return the strip-wide band along one edge of an image.

    edge in {left, right, top, bottom}; strip is width in pixels.
    """
    h, w = img.shape[:2]
    if edge == "left":
        return img[:, :strip]
    if edge == "right":
        return img[:, max(0, w - strip):w]
    if edge == "top":
        return img[:strip, :]
    if edge == "bottom":
        return img[max(0, h - strip):h, :]
    raise ValueError(f"Unknown edge: {edge}")


def edge_crop_origin(shape, edge: str, strip: int):
    """Top-left (x, y) of the edge strip inside the full tile.

    Needed to convert a strip-local shift into a full-tile offset.
    """
    h, w = shape[:2]
    strip_x = min(strip, w)
    strip_y = min(strip, h)
    if edge == "left":
        return np.array([0.0, 0.0])
    if edge == "right":
        return np.array([float(w - strip_x), 0.0])
    if edge == "top":
        return np.array([0.0, 0.0])
    if edge == "bottom":
        return np.array([0.0, float(h - strip_y)])
    raise ValueError(f"Unknown edge: {edge}")


# ---------------------------------------------------------------------------
# Translation RANSAC (shared by SIFT and RoMa matchers)
# ---------------------------------------------------------------------------
def translation_ransac(pts_a, pts_b, threshold):
    """Find the dominant pure-translation shift between matched point sets.

    Each candidate shift = pts_b - pts_a votes; the shift with the most
    inliers (within `threshold` px) wins, then the inlier median is returned.

    Returns (shift[2], inlier_mask) or (None, None) if no points.
    """
    shifts = pts_b - pts_a
    best_inliers = None
    best_count = 0
    for shift in shifts:
        errors = np.linalg.norm(shifts - shift, axis=1)
        inliers = errors < threshold
        count = int(np.sum(inliers))
        if count > best_count:
            best_count = count
            best_inliers = inliers
    if best_inliers is None:
        return None, None
    final_shift = np.median(shifts[best_inliers], axis=0)
    return final_shift, best_inliers
