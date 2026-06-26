import json
import re
import shutil
from pathlib import Path
from collections import OrderedDict
from romav2 import RoMaV2
import torch
import time
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Initialize the model on the GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading RoMaV2 on {device}...")
roma_model = RoMaV2().to(device)

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")


# Choose one:
# "cytology-small"
# "cytology-large"
# "histology-small"
# "histology-large"

DATASET_NAME = "histology-small"

VALID_DATASETS = {
    "cytology-small",
    "cytology-large",
    "histology-small",
    "histology-large",
}

if DATASET_NAME not in VALID_DATASETS:
    raise ValueError(
        f"Unknown dataset: {DATASET_NAME}. Choose one of: {sorted(VALID_DATASETS)}"
    )

BASE_DIR = Path.cwd()

BASE_INPUT_ROOT = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

INPUT_FOLDER = BASE_INPUT_ROOT / DATASET_NAME / "Images-FOV"

GRAPH_PATH = OUTPUT_DIR / "stitch_graph.json"
TRANSFORMS_PATH = OUTPUT_DIR / "transforms_translation.json"
DEBUG_DIR = OUTPUT_DIR / "debug_matches"

print("Selected dataset:", DATASET_NAME)
print("Input folder:", INPUT_FOLDER)
print("Output folder:", OUTPUT_DIR)

# RESET OUTPUT FOLDER - RUN IF YOU NEED TO


def reset_output_folder(output_dir):
    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Clean output folder ready: {output_dir}")


reset_output_folder(OUTPUT_DIR)


# BORDERS JSON FUNCTIONS

TILE_RE = re.compile(r"^(\d+)_(-?\d+)_(-?\d+)\.bmp$", re.IGNORECASE)

OPPOSITE = {
    "left": "right",
    "right": "left",
    "top": "bottom",
    "bottom": "top",
}


def load_tiles(folder):
    tiles = []

    if not folder.exists():
        raise FileNotFoundError(f"Missing input folder: {folder}")

    for file in sorted(folder.iterdir()):
        if not file.is_file():
            continue

        match = TILE_RE.match(file.name)

        if match:
            tile_id = int(match.group(1))
            x = int(match.group(2))
            y = int(match.group(3))

            tiles.append(
                {
                    "id": tile_id,
                    "x": x,
                    "y": y,
                    "file": str(file),
                }
            )

    return tiles


def build_graph(tiles):
    if not tiles:
        return OrderedDict()

    xs = sorted(set(tile["x"] for tile in tiles))
    ys = sorted(set(tile["y"] for tile in tiles))

    coord_to_tile = {(tile["x"], tile["y"]): tile for tile in tiles}

    graph = OrderedDict()

    sorted_tiles = sorted(tiles, key=lambda t: (ys.index(t["y"]), xs.index(t["x"])))

    for tile in sorted_tiles:
        tile_id = tile["id"]
        x = tile["x"]
        y = tile["y"]

        col = xs.index(x)
        row = ys.index(y)

        neighbors = [
            ("right", col + 1, row),
            ("bottom", col, row + 1),
        ]

        for edge, neighbor_col, neighbor_row in neighbors:
            if neighbor_col >= len(xs) or neighbor_row >= len(ys):
                continue

            neighbor_coord = (xs[neighbor_col], ys[neighbor_row])
            neighbor = coord_to_tile.get(neighbor_coord)

            if neighbor is None:
                continue

            neighbor_id = neighbor["id"]
            opposite_edge = OPPOSITE[edge]

            key = f"connection_{tile_id}_{neighbor_id}"

            graph[key] = {
                "label": f"slide_{tile_id}:{edge} <-> slide_{neighbor_id}:{opposite_edge}",
                "type": "vertical" if edge == "right" else "horizontal",
                "a": {
                    "slide": tile_id,
                    "edge": edge,
                    "file": tile["file"],
                    "coord": [x, y],
                },
                "b": {
                    "slide": neighbor_id,
                    "edge": opposite_edge,
                    "file": neighbor["file"],
                    "coord": [neighbor["x"], neighbor["y"]],
                },
            }

    return graph


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved: {path}")


# BORDERS JSON

tiles = load_tiles(INPUT_FOLDER)

print(f"Tiles found: {len(tiles)}")

if not tiles:
    raise RuntimeError(f"No tiles found in: {INPUT_FOLDER}")

graph = build_graph(tiles)

print(f"Connections found: {len(graph)}")

save_json(graph, GRAPH_PATH)

# HELPER FUNCTIONS

EDGE_STRIP = 50
RATIO_TEST = 0.75
RANSAC_THRESHOLD = 3.0
MAX_MATCHES_TO_DRAW = 100


def crop_edge(img, edge, strip):
    h, w = img.shape[:2]

    if edge == "left":
        return img[:, :strip]

    if edge == "right":
        return img[:, w - strip : w]

    if edge == "top":
        return img[:strip, :]

    if edge == "bottom":
        return img[h - strip : h, :]

    raise ValueError(f"Unknown edge: {edge}")


def translation_ransac(pts_a, pts_b, threshold):
    shifts = pts_b - pts_a

    best_inliers = None
    best_count = 0

    for shift in shifts:
        errors = np.linalg.norm(shifts - shift, axis=1)
        inliers = errors < threshold
        count = np.sum(inliers)

        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None:
        return None, None

    final_shift = np.median(shifts[best_inliers], axis=0)

    return final_shift, best_inliers


# CONNECTIONS WITH ROMAV2


def process_connection_roma(name, conn, edge_strip=EDGE_STRIP):

    # RoMaV2 expects RGB images, not grayscale
    img_a = cv2.imread(conn["a"]["file"])
    img_b = cv2.imread(conn["b"]["file"])

    if img_a is None or img_b is None:
        return {
            "status": "failed",
            "reason": "could not load image",
            "label": conn.get("label"),
        }

    img_a_rgb = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
    img_b_rgb = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)

    # Crop edges (using your existing crop_edge function)
    strip_a = crop_edge(img_a_rgb, conn["a"]["edge"], edge_strip)
    strip_b = crop_edge(img_b_rgb, conn["b"]["edge"], edge_strip)

    H_A, W_A = strip_a.shape[:2]
    H_B, W_B = strip_b.shape[:2]

    # 2. Run Dense Matching
    with torch.inference_mode():
        # Predict dense warp
        preds = roma_model.match(strip_a, strip_b)

        # Sample points from the dense prediction
        # 2000 is a good balance of speed/accuracy. You can increase this to 5000 if needed.
        matches, overlaps, precision_AB, precision_BA = roma_model.sample(preds, 2000)

        # Convert RoMa's [-1,1] coordinates back to pixel space
        kptsA, kptsB = roma_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)

    # Convert to CPU numpy arrays for RANSAC
    pts_a = kptsA.cpu().numpy()
    pts_b = kptsB.cpu().numpy()

    if len(pts_a) < 4:
        return {
            "status": "failed",
            "reason": "too few matches",
            "label": conn.get("label"),
            "matches": len(pts_a),
        }

    # 3. Filter with your Translation RANSAC
    shift, inlier_mask = translation_ransac(pts_a, pts_b, RANSAC_THRESHOLD)

    if shift is None:
        return {
            "status": "failed",
            "reason": "ransac failed",
            "label": conn.get("label"),
            "matches": len(pts_a),
        }

    inlier_count = np.sum(inlier_mask)
    inlier_ratio = inlier_count / len(pts_a)
    dx, dy = shift

    H = [
        [1.0, 0.0, float(dx)],
        [0.0, 1.0, float(dy)],
        [0.0, 0.0, 1.0],
    ]

    # 4. Visualization
    # Convert back to BGR for cv2 drawing
    strip_a_bgr = cv2.cvtColor(strip_a, cv2.COLOR_RGB2BGR)
    strip_b_bgr = cv2.cvtColor(strip_b, cv2.COLOR_RGB2BGR)

    # Mock OpenCV KeyPoints and DMatches so we can use cv2.drawMatches
    cv_kp_a = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1) for pt in pts_a]
    cv_kp_b = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1) for pt in pts_b]
    cv_matches = [
        cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0) for i in range(len(pts_a))
    ]

    inlier_matches = [m for m, keep in zip(cv_matches, inlier_mask) if keep]

    vis = cv2.drawMatches(
        strip_a_bgr,
        cv_kp_a,
        strip_b_bgr,
        cv_kp_b,
        inlier_matches[:MAX_MATCHES_TO_DRAW],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    debug_path = DEBUG_DIR / f"{name}_roma.png"
    cv2.imwrite(str(debug_path), vis)

    # Note the threshold changes here:
    # Dense matching samples the *whole* strip uniformly. A lot of those samples
    # fall on non-overlapping regions, so inlier_ratio will be much lower than SIFT.
    status = "good" if inlier_count >= 15 and inlier_ratio >= 0.1 else "suspicious"

    return {
        "status": status,
        "label": conn["label"],
        "type": conn["type"],
        "a": conn["a"],
        "b": conn["b"],
        "edge_strip": edge_strip,
        "matches": len(pts_a),
        "inliers": int(inlier_count),
        "inlier_ratio": round(float(inlier_ratio), 3),
        "dx": float(dx),
        "dy": float(dy),
        "transform_a_to_b": H,
        "debug_image": str(debug_path),
    }


# CREATE THE TRANSFORMATIONS JSON

with open(GRAPH_PATH, "r") as f:
    graph = json.load(f)

results = {}

for name, conn in graph.items():
    print(f"\nProcessing {name}: {conn['label']}")

    result = process_connection_roma(name, conn)
    results[name] = result

    print("Status:", result.get("status"))
    print("Reason:", result.get("reason"))
    print("Matches:", result.get("matches"))
    print("Inliers:", result.get("inliers"))
    print("Inlier ratio:", result.get("inlier_ratio"))
    print("dx, dy:", result.get("dx"), result.get("dy"))

save_json(results, TRANSFORMS_PATH)

print("\nDone.")
print(f"Graph saved to: {GRAPH_PATH}")
print(f"Transforms saved to: {TRANSFORMS_PATH}")
print(f"Debug images saved to: {DEBUG_DIR}")

# SUMMARY OF GOOD, FAILED, SUSPICIOUS BORDERS

good = sum(1 for r in results.values() if r.get("status") == "good")
suspicious = sum(1 for r in results.values() if r.get("status") == "suspicious")
failed = sum(1 for r in results.values() if r.get("status") == "failed")

print("Summary")
print("-------")
print("Dataset:", DATASET_NAME)
print("Tiles:", len(tiles))
print("Connections:", len(graph))
print("Good:", good)
print("Suspicious:", suspicious)
print("Failed:", failed)
