# Slide-Stitching Pipeline

A developer-ready, script-based pipeline for stitching microscope slide tiles
(CellaVision pathology FOV images) into a single mosaic.

This repository is the productionized version of the original Colab notebooks.
Each notebook cell becomes a **standalone, ordered script**; outputs keep the
**same filenames** the notebooks produced, so existing tools and notebooks
continue to work unchanged.

> **Status:** Stages 1–3 (borders, cutting, matching) and stage 7 (stitching)
> are implemented and tested. The matcher supports **two switchable methods**
> (SIFT, RoMa, hybrid). Stitching produces a pyramidal OME-TIFF (flat PNG removed for speed).
> Intermediate stages (metrics, spanning tree, global optimization, error
> estimation) plug into the same ordered structure.

---

## Quick start

```bash
# 1. Run the setup script to build your environment and install dependencies
# (This automatically detects your CUDA runtime, sets up PyTorch, and configures RoMaV2)
# See the setup.sh file for more usage info
./setup.sh --venv .venv

# 1a. Activate the virtual environment
source .venv/bin/activate

# (optional) for the RoMa / hybrid matchers on a GPU, also:
#   - install a CUDA-matched torch build, then
#   pip install -r requirements-roma.txt
# see requirements-roma.txt for the exact steps.

# 2. point the pipeline at your data and run everything
DATASET_NAME=cytology-small \
INPUT_ROOT=/path/to/CellaVision/Pathology \
OUTPUT_DIR=./output \
python run_all.py
```

Or run a single stage on its own:

```bash
python scripts/01_build_borders.py
python scripts/02_cut_borders.py
```

---

## Expected data layout

Local filesystem. The pipeline reads tiles from:

```
<INPUT_ROOT>/<DATASET_NAME>/Images-FOV/<id>_<x>_<y>.bmp
```

- `<id>` — integer tile id
- `<x>`,`<y>` — integer stage coordinates (may be negative)

Example: `42_1000_2000.bmp` is tile 42 at stage position (1000, 2000).

Valid datasets: `cytology-small`, `cytology-large`, `histology-small`,
`histology-large`.

---

## Configuration

All settings live in **`config.py`**. You can override the important ones via
environment variables without editing the file:

| Variable       | Meaning                         | Default          |
| -------------- | ------------------------------- | ---------------- |
| `DATASET_NAME` | which dataset to process        | `cytology-small` |
| `INPUT_ROOT`   | root folder containing datasets | `./data`         |
| `OUTPUT_DIR`   | where artifacts are written     | `./output`       |

Key algorithm parameters (in `config.py`):

| Name               | Meaning                                            | Default |
| ------------------ | -------------------------------------------------- | ------- |
| `EDGE_STRIP`       | width (px) of the border strip cut from each edge  | `150`   |
| `RATIO_TEST`       | Lowe's ratio-test threshold for SIFT matches       | `0.75`  |
| `RANSAC_THRESHOLD` | inlier distance threshold (px) for translation fit | `3.0`   |

---

## Pipeline stages

Scripts are numbered so the run order is obvious. Each is independently
runnable and reads/writes canonical files in `OUTPUT_DIR`.

### Stage 1 — `01_build_borders.py`

Discovers tiles and builds the **borders graph**: every shared border between
two adjacent tiles becomes one connection. Each tile links to its right and
bottom neighbor, so each border is represented exactly once.

- **Reads:** `<INPUT_ROOT>/<DATASET_NAME>/Images-FOV/*.bmp`
- **Writes:** `output/stitch_graph.json`

Each connection records which edge of each tile forms the border:

```json
"connection_0_3": {
  "label": "slide_0:right <-> slide_3:left",
  "type": "vertical",
  "a": { "slide": 0, "edge": "right",  "file": "...0_0_0.bmp",    "coord": [0, 0] },
  "b": { "slide": 3, "edge": "left",   "file": "...3_1000_0.bmp", "coord": [1000, 0] }
}
```

`type` is `vertical` for left–right neighbors and `horizontal` for top–bottom.

### Stage 2 — `02_cut_borders.py`

**Cuts** the `EDGE_STRIP`-wide band along each shared border from both tiles —
exactly the regions the matcher will compare. Cutting them to disk makes the
matching stage simpler and lets you visually inspect every border.

- **Reads:** `output/stitch_graph.json` + the tile images
- **Writes:**
    - `output/border_strips/<connection>__a_<edge>.png`
    - `output/border_strips/<connection>__b_<edge>.png`
    - `output/border_strips/strips_index.json`

`strips_index.json` maps each connection to its two strip files and records
each strip's size, so later stages can load strips directly.

### Stage 3 — `03_match.py` _(switchable method)_

Computes the per-border translation `(dx, dy)` between adjacent tiles, plus an
`inlier_ratio` quality score and a `good`/`suspicious`/`failed` status. **You
choose the matching method:**

| `MATCH_METHOD`   | How it works                                             | Needs          |
| ---------------- | -------------------------------------------------------- | -------------- |
| `sift` (default) | SIFT keypoints + Lowe ratio test + translation RANSAC    | CPU only       |
| `roma`           | RoMaV2 dense neural matcher + translation RANSAC         | GPU + `romav2` |
| `hybrid`         | SIFT first; fall back to RoMa where SIFT isn't confident | GPU + `romav2` |

Both write the **identical schema**, so everything downstream is method-agnostic.

```bash
MATCH_METHOD=sift python scripts/03_match.py      # default, CPU
MATCH_METHOD=roma python scripts/03_match.py      # GPU dense matching
MATCH_METHOD=hybrid python scripts/03_match.py    # SIFT, RoMa fallback
```

- **Reads:** `stitch_graph.json` + tiles
- **Writes:** `transforms_translation.json`, `debug_matches/<connection>.png`

> RoMa samples the whole strip uniformly, so its `inlier_ratio` is naturally
> lower than SIFT's — the "good" threshold for RoMa is relaxed accordingly
> (`ROMA_GOOD_MIN_INLIER_RATIO` in `config.py`).

> **Hybrid** runs SIFT on every border and keeps its result only when it is
> _confidently_ good (`HYBRID_SIFT_MIN_INLIERS` inliers **and**
> `HYBRID_SIFT_MIN_INLIER_RATIO` ratio). Borders that don't clear that bar —
> typically low-texture / blank backgrounds where SIFT struggles — fall back to
> RoMa. You get SIFT's speed on easy borders and only pay the GPU cost on the
> hard ones. Each result records which backend won (`method`) and why it fell
> back (`hybrid_note`); the stage prints a per-backend breakdown.

### Stage 7 — `07_stitch.py` _(PNG + pyramidal OME-TIFF)_

Assembles the mosaic. Places `good` connections via metadata-scaled BFS, then
fills all-failed tiles from their neighbors so no tile is dropped. Produces:

- a **pyramidal OME-TIFF** (`stitched_pyramid.ome.tif`) — multi-resolution and
  viewer-friendly (QuPath, napari, OMERO), rendered **one level at a time into
  disk-backed memmaps** so even gigapixel mosaics don't blow up RAM;
- the **layout JSON** (`stitch_layout.json`) recording every tile's position.

```bash
python scripts/07_stitch.py
PYRAMID_COMPRESSION=jpeg python scripts/07_stitch.py   # smaller, lossy
```

Pyramid settings (`config.py`): `PYRAMID_TILE_SIZE` (512), `PYRAMID_MIN_SIZE`
(1024 — stop building levels below this), `PYRAMID_COMPRESSION` (`deflate`
lossless / `jpeg` lossy).

#### Overlap blending

Where tiles overlap, `BLEND_MODE` controls how they're combined — in **both**
the PNG and the pyramidal OME-TIFF:

| `BLEND_MODE`        | Behavior                                                                                                  |
| ------------------- | --------------------------------------------------------------------------------------------------------- |
| `none`              | last tile wins (hard overwrite; fast, visible seams)                                                      |
| `average`           | mean of all tiles covering a pixel (removes seams, can be soft)                                           |
| `feather` (default) | distance-weighted blend — each tile fades out toward its edges so seams disappear smoothly (best quality) |

```bash
BLEND_MODE=feather python scripts/07_stitch.py    # default, smoothest
BLEND_MODE=average python scripts/07_stitch.py
BLEND_MODE=none    python scripts/07_stitch.py    # fastest, sharp seams
```

`FEATHER_WIDTH` (px) sets the ramp width at each tile edge; it auto-scales per
pyramid level. Blending uses **disk-backed float accumulators** in the pyramid
path, so even gigapixel mosaics blend without exhausting RAM. The chosen mode
is recorded in `stitch_layout.json` (`stitch_info.blend_mode`).

> **Note on placement source:** stage 7 uses the matcher transforms directly
> (Gustaf's metadata-scaled layout). If you run the global-optimization stage,
> point the stitcher at `global_positions_optimized_<metric>.json` instead —
> the layout dict format is the same.

---

## Output files (canonical names)

These names match the original notebooks so downstream code keeps working:

| File                             | Produced by | Contents                                |
| -------------------------------- | ----------- | --------------------------------------- |
| `stitch_graph.json`              | stage 1     | the borders/neighbor graph              |
| `border_strips/`                 | stage 2     | cut edge strips + `strips_index.json`   |
| `transforms_translation.json`    | stage 3\*   | per-border SIFT/RANSAC shifts           |
| `metrics.json` / `metrics.csv`   | stage 4\*   | NCC/SSIM/LPIPS per border               |
| `global_positions_<metric>.json` | stage 5\*   | spanning-tree global layout             |
| `global_positions_optimized_*`   | stage 6\*   | least-squares global layout + residuals |
| `stitched_pyramid.ome.tif`       | stage 7     | final mosaic (multi-resolution)         |

\* later stages, same ordered structure.

---

## Project structure

```
stitch_pipeline/
├── config.py            # all paths + parameters (one place)
├── common.py            # shared helpers + auto dependency install
├── matchers.py          # SIFT + RoMa matching backends (stage 3)
├── stitcher.py          # layout + PNG + pyramidal OME-TIFF (stage 7)
├── requirements.txt     # dependencies (scripts also self-install)
├── run_all.py           # orchestrator: runs stages in order
├── scripts/
│   ├── 01_build_borders.py   # discover tiles → borders graph
│   ├── 02_cut_borders.py     # cut the edge strips to be matched
│   ├── 03_match.py           # SIFT / RoMa → transforms (switchable)
│   └── 07_stitch.py          # PNG + pyramidal OME-TIFF
└── output/              # all artifacts land here
```

### Design notes

- **Self-installing:** importing `common` runs `ensure_requirements()`, which
  pip-installs any missing core packages. Stage-specific heavy deps
  (torch/lpips/scipy) install on demand in the stage that needs them.
- **One config:** every path and parameter is in `config.py`; scripts never
  hardcode paths.
- **Stable keys:** every stage keys its results by the connection name
  (`connection_<a>_<b>`) from stage 1, so results join cleanly across stages.
- **Independently runnable:** the orchestrator just shells out to each
  numbered script, so you can always run/debug a single stage alone.

---

## Running a subset

```bash
python run_all.py --only 1      # just stage 1
python run_all.py --from 2      # stage 2 onward
python run_all.py --to 1        # up to and including stage 1
```
