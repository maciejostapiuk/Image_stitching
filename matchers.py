"""
matchers.py
===========
Feature-matching backends for stage 3. Each backend takes a connection and
returns a per-border result dict in the SAME schema (so transforms_translation
.json is identical regardless of method):

    {status, label, type, a, b, edge_strip,
     matches, inliers, inlier_ratio, dx, dy, transform_a_to_b, debug_image}

Two backends:
    match_sift(...)  - classic SIFT keypoints + Lowe ratio test + translation RANSAC
    match_roma(...)  - RoMaV2 dense neural matcher + translation RANSAC

Both reuse common.translation_ransac so the geometry is identical; only the
correspondence-finding differs.
"""

from __future__ import annotations

from pathlib import Path

import config
import common
from common import crop_edge, translation_ransac, read_color, read_gray

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Shared: build the result dict
# ---------------------------------------------------------------------------
def _make_result(
    conn,
    edge_strip,
    pts_a,
    shift,
    inlier_mask,
    good_min_inliers,
    good_min_ratio,
    debug_path,
):
    inlier_count = int(np.sum(inlier_mask))
    inlier_ratio = inlier_count / len(pts_a)
    dx, dy = float(shift[0]), float(shift[1])
    H = [[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]]
    status = (
        "good"
        if inlier_count >= good_min_inliers and inlier_ratio >= good_min_ratio
        else "suspicious"
    )
    return {
        "status": status,
        "label": conn["label"],
        "type": conn["type"],
        "a": conn["a"],
        "b": conn["b"],
        "edge_strip": edge_strip,
        "matches": int(len(pts_a)),
        "inliers": inlier_count,
        "inlier_ratio": round(float(inlier_ratio), 3),
        "dx": dx,
        "dy": dy,
        "transform_a_to_b": H,
        "debug_image": str(debug_path) if debug_path else None,
    }


# ---------------------------------------------------------------------------
# SIFT backend
# ---------------------------------------------------------------------------
def match_sift(name, conn, edge_strip=None, debug_dir=None):
    edge_strip = edge_strip if edge_strip is not None else config.EDGE_STRIP

    img_a = read_gray(conn["a"]["file"])
    img_b = read_gray(conn["b"]["file"])
    if img_a is None or img_b is None:
        return {
            "status": "failed",
            "reason": "could not load image",
            "label": conn.get("label"),
        }

    strip_a = crop_edge(img_a, conn["a"]["edge"], edge_strip)
    strip_b = crop_edge(img_b, conn["b"]["edge"], edge_strip)

    sift = cv2.SIFT_create()
    kp_a, des_a = sift.detectAndCompute(
        np.ascontiguousarray(strip_a, dtype=np.uint8), None
    )
    kp_b, des_b = sift.detectAndCompute(
        np.ascontiguousarray(strip_b, dtype=np.uint8), None
    )
    if des_a is None or des_b is None:
        return {
            "status": "failed",
            "reason": "no descriptors",
            "label": conn.get("label"),
            "features_a": len(kp_a) if kp_a else 0,
            "features_b": len(kp_b) if kp_b else 0,
        }

    des_a = np.asarray(des_a, dtype=np.float32)
    des_b = np.asarray(des_b, dtype=np.float32)
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_a, des_b, k=2)
    matches = [
        m
        for m, n in (p for p in knn if len(p) == 2)
        if m.distance < config.RATIO_TEST * n.distance
    ]
    if len(matches) < 4:
        return {
            "status": "failed",
            "reason": "too few matches",
            "label": conn.get("label"),
            "features_a": len(kp_a),
            "features_b": len(kp_b),
            "matches": len(matches),
        }

    pts_a = np.float32([kp_a[m.queryIdx].pt for m in matches])
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in matches])
    shift, inlier_mask = translation_ransac(pts_a, pts_b, config.RANSAC_THRESHOLD)
    if shift is None:
        return {
            "status": "failed",
            "reason": "ransac failed",
            "label": conn.get("label"),
            "features_a": len(kp_a),
            "features_b": len(kp_b),
            "matches": len(matches),
        }

    debug_path = None
    if debug_dir is not None:
        inlier_matches = [m for m, keep in zip(matches, inlier_mask) if keep]
        vis = cv2.drawMatches(
            strip_a,
            kp_a,
            strip_b,
            kp_b,
            inlier_matches[: config.MAX_MATCHES_TO_DRAW],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        debug_path = Path(debug_dir) / f"{name}.png"
        cv2.imwrite(str(debug_path), vis)

    return _make_result(
        conn,
        edge_strip,
        pts_a,
        shift,
        inlier_mask,
        config.GOOD_MIN_INLIERS,
        config.GOOD_MIN_INLIER_RATIO,
        debug_path,
    )


# ---------------------------------------------------------------------------
# RoMa backend (lazy model load so SIFT users never need torch/romav2)
# ---------------------------------------------------------------------------
_roma_model = None
_roma_device = None


def _load_roma():
    global _roma_model, _roma_device
    if _roma_model is not None:
        return _roma_model, _roma_device
    common.ensure_requirements({"torch": "torch"})
    import torch

    try:
        from romav2 import RoMaV2
    except ImportError as e:
        raise ImportError(
            "RoMaV2 not installed. The 'roma'/'hybrid' methods need the romav2 "
            "package (and ideally a CUDA GPU). See requirements-roma.txt for "
            "install steps, or use MATCH_METHOD=sift (CPU, no extra deps)."
        ) from e
    _roma_device = "cuda" if torch.cuda.is_available() else "cpu"
    common.log(f"[roma] loading RoMaV2 on {_roma_device}...")
    _roma_model = RoMaV2().to(_roma_device)
    return _roma_model, _roma_device


def match_roma(name, conn, edge_strip=None, debug_dir=None):
    edge_strip = edge_strip if edge_strip is not None else config.EDGE_STRIP
    import torch

    model, _ = _load_roma()

    img_a = read_color(conn["a"]["file"])
    img_b = read_color(conn["b"]["file"])
    if img_a is None or img_b is None:
        return {
            "status": "failed",
            "reason": "could not load image",
            "label": conn.get("label"),
        }

    strip_a = crop_edge(
        cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB), conn["a"]["edge"], edge_strip
    )
    strip_b = crop_edge(
        cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB), conn["b"]["edge"], edge_strip
    )
    H_A, W_A = strip_a.shape[:2]
    H_B, W_B = strip_b.shape[:2]

    with torch.inference_mode():
        preds = model.match(strip_a, strip_b)
        matches, overlaps, prec_ab, prec_ba = model.sample(preds, config.ROMA_SAMPLES)
        kptsA, kptsB = model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
    pts_a = kptsA.cpu().numpy()
    pts_b = kptsB.cpu().numpy()

    if len(pts_a) < 4:
        return {
            "status": "failed",
            "reason": "too few matches",
            "label": conn.get("label"),
            "matches": len(pts_a),
        }

    shift, inlier_mask = translation_ransac(pts_a, pts_b, config.RANSAC_THRESHOLD)
    if shift is None:
        return {
            "status": "failed",
            "reason": "ransac failed",
            "label": conn.get("label"),
            "matches": len(pts_a),
        }

    debug_path = None
    if debug_dir is not None:
        strip_a_bgr = cv2.cvtColor(strip_a, cv2.COLOR_RGB2BGR)
        strip_b_bgr = cv2.cvtColor(strip_b, cv2.COLOR_RGB2BGR)
        cv_kp_a = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in pts_a]
        cv_kp_b = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in pts_b]
        cv_matches = [cv2.DMatch(i, i, 0) for i in range(len(pts_a))]
        inlier_matches = [m for m, keep in zip(cv_matches, inlier_mask) if keep]
        vis = cv2.drawMatches(
            strip_a_bgr,
            cv_kp_a,
            strip_b_bgr,
            cv_kp_b,
            inlier_matches[: config.MAX_MATCHES_TO_DRAW],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        debug_path = Path(debug_dir) / f"{name}_roma.png"
        cv2.imwrite(str(debug_path), vis)

    return _make_result(
        conn,
        edge_strip,
        pts_a,
        shift,
        inlier_mask,
        config.ROMA_GOOD_MIN_INLIERS,
        config.ROMA_GOOD_MIN_INLIER_RATIO,
        debug_path,
    )


# ---------------------------------------------------------------------------
# Hybrid backend: SIFT first, fall back to RoMa when SIFT isn't confident
# ---------------------------------------------------------------------------
def _sift_is_confident(result) -> bool:
    """True only when SIFT produced a confidently-good match."""
    if result.get("status") != "good" and "dx" not in result:
        return False
    if "inliers" not in result or "inlier_ratio" not in result:
        return False
    return (
        result["inliers"] >= config.HYBRID_SIFT_MIN_INLIERS
        and result["inlier_ratio"] >= config.HYBRID_SIFT_MIN_INLIER_RATIO
    )


def match_hybrid(name, conn, edge_strip=None, debug_dir=None):
    """Try SIFT first; if it isn't confidently good, fall back to RoMa.

    SIFT is fast and accurate on textured borders. On low-texture / blank
    backgrounds it fails or returns weak matches — exactly where RoMa's dense
    matching shines. The hybrid keeps SIFT's strong matches and only pays the
    RoMa (GPU) cost on the hard borders.

    A `method` field is added to the result so you can see which matcher won.
    """
    sift_res = match_sift(name, conn, edge_strip=edge_strip, debug_dir=debug_dir)

    if _sift_is_confident(sift_res):
        sift_res["method"] = "sift"
        return sift_res

    # SIFT wasn't confident -> try RoMa
    sift_reason = sift_res.get("reason") or (
        f"weak sift (inliers={sift_res.get('inliers')}, "
        f"ratio={sift_res.get('inlier_ratio')})"
    )
    roma_res = match_roma(name, conn, edge_strip=edge_strip, debug_dir=debug_dir)

    # If RoMa also failed, keep whichever has a usable shift; prefer SIFT's
    # info for the failure record so the reason chain is visible.
    if "dx" not in roma_res:
        # both failed (or roma failed). Return SIFT result if it had a shift.
        if "dx" in sift_res:
            sift_res["method"] = "sift"
            sift_res["hybrid_note"] = "roma fallback also failed; kept sift"
            return sift_res
        roma_res["method"] = "roma"
        roma_res["hybrid_note"] = f"sift first failed ({sift_reason})"
        return roma_res

    roma_res["method"] = "roma"
    roma_res["hybrid_note"] = f"fell back from sift ({sift_reason})"
    return roma_res


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
def get_matcher(method: str):
    method = method.lower()
    if method == "sift":
        return match_sift
    if method == "roma":
        return match_roma
    if method == "hybrid":
        return match_hybrid
    raise ValueError(
        f"Unknown match method {method!r}. "
        f"Choose one of: {sorted(config.VALID_MATCH_METHODS)}"
    )
