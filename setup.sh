#!/usr/bin/env bash
# =============================================================================
# setup.sh  -  install the stitching pipeline's dependencies, with CUDA support
# =============================================================================
# Installs:
#   1. CUDA-enabled PyTorch matched to your detected CUDA toolkit (or CPU build
#      if no GPU / --cpu is given)
#   2. the base pipeline requirements (numpy, opencv, scikit-image, tifffile, scipy)
#   3. (optional) RoMaV2 for the roma/hybrid matchers
#
# Usage:
#   ./setup.sh                 # auto-detect CUDA, install everything + RoMa
#   ./setup.sh --cpu           # force CPU-only torch (no GPU)
#   ./setup.sh --cuda 121      # force a specific CUDA build (e.g. 12.1 -> 121)
#   ./setup.sh --no-roma       # skip RoMaV2 (SIFT-only setup)
#   ./setup.sh --venv .venv    # create/use a virtualenv at .venv first
#
# Notes:
#   - Works on Linux and Windows (Git Bash / WSL). macOS has no CUDA: it falls
#     back to the CPU/MPS torch build automatically.
#   - Re-runnable: pip will skip already-satisfied packages.
# =============================================================================

set -euo pipefail

# ---- defaults ---------------------------------------------------------------
FORCE_CPU=0
FORCE_CUDA=""          # e.g. "121", "118", "124"
INSTALL_ROMA=1
VENV_PATH=""
PYTHON="${PYTHON:-python3}"

# ---- parse args -------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu)      FORCE_CPU=1; shift ;;
    --cuda)     FORCE_CUDA="${2:-}"; shift 2 ;;
    --no-roma)  INSTALL_ROMA=0; shift ;;
    --venv)     VENV_PATH="${2:-.venv}"; shift 2 ;;
    --python)   PYTHON="${2}"; shift 2 ;;
    -h|--help)
      sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

echo "==============================================================="
echo " Stitching pipeline setup"
echo "==============================================================="

# ---- optional virtualenv ----------------------------------------------------
if [[ -n "$VENV_PATH" ]]; then
  echo "[venv] creating/using virtualenv at: $VENV_PATH"
  "$PYTHON" -m venv "$VENV_PATH"
  # shellcheck disable=SC1091
  if [[ -f "$VENV_PATH/bin/activate" ]]; then
    source "$VENV_PATH/bin/activate"
  else
    source "$VENV_PATH/Scripts/activate"   # Git Bash on Windows
  fi
  PYTHON="python"
fi

PIP="$PYTHON -m pip"
echo "[pip] upgrading pip..."
$PIP install --upgrade pip >/dev/null

# ---- detect CUDA version ----------------------------------------------------
# Maps a detected CUDA toolkit version to a PyTorch wheel index tag.
# PyTorch publishes builds for a limited set of CUDA versions; we snap to the
# nearest supported one.
detect_cuda_tag() {
  local raw=""
  # Prefer nvcc, fall back to nvidia-smi's reported CUDA version.
  if command -v nvcc >/dev/null 2>&1; then
    raw="$(nvcc --version 2>/dev/null | grep -oE 'release [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -1)"
  fi
  if [[ -z "$raw" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    raw="$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -1)"
  fi
  if [[ -z "$raw" ]]; then
    echo ""        # no CUDA found
    return
  fi

  local major minor num
  major="${raw%%.*}"
  minor="${raw##*.}"
  num=$((major * 10 + minor))

  # Snap to a PyTorch-supported CUDA wheel tag.
  if   (( num >= 124 )); then echo "cu124"
  elif (( num >= 121 )); then echo "cu121"
  elif (( num >= 118 )); then echo "cu118"
  else                        echo "cu118"   # oldest commonly-supported
  fi
}

OS="$(uname -s 2>/dev/null || echo unknown)"
CUDA_TAG=""

if [[ "$FORCE_CPU" -eq 1 ]]; then
  echo "[torch] --cpu given: installing CPU-only build."
  CUDA_TAG="cpu"
elif [[ -n "$FORCE_CUDA" ]]; then
  CUDA_TAG="cu${FORCE_CUDA}"
  echo "[torch] forced CUDA build: $CUDA_TAG"
elif [[ "$OS" == "Darwin" ]]; then
  echo "[torch] macOS detected: no CUDA, using default (CPU/MPS) build."
  CUDA_TAG="default"
else
  CUDA_TAG="$(detect_cuda_tag)"
  if [[ -z "$CUDA_TAG" ]]; then
    echo "[torch] no NVIDIA GPU / CUDA detected: installing CPU-only build."
    CUDA_TAG="cpu"
  else
    echo "[torch] detected CUDA -> wheel tag: $CUDA_TAG"
  fi
fi

# ---- install torch ----------------------------------------------------------
case "$CUDA_TAG" in
  cpu)
    $PIP install "torch>=2.0" --index-url https://download.pytorch.org/whl/cpu
    ;;
  default)
    # macOS / let pip pick the platform default
    $PIP install "torch>=2.0"
    ;;
  cu*)
    echo "[torch] installing CUDA build from PyTorch index ($CUDA_TAG)..."
    $PIP install "torch>=2.0" --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
    ;;
esac

# ---- install the rest of the base requirements ------------------------------
# (torch is already handled above; install the others explicitly so we don't
#  re-resolve torch from the default index.)
echo "[deps] installing base pipeline requirements..."
$PIP install \
  "numpy>=1.24" \
  "opencv-python>=4.8" \
  "scikit-image>=0.21" \
  "tifffile>=2023.7" \
  "scipy>=1.10"

# ---- optional: RoMaV2 -------------------------------------------------------
if [[ "$INSTALL_ROMA" -eq 1 && "$CUDA_TAG" != "cpu" ]]; then
  echo "[roma] installing RoMaV2 (for MATCH_METHOD=roma|hybrid)..."
  $PIP install "git+https://github.com/Parskatt/RoMa.git" || {
    echo "[roma] WARNING: RoMaV2 install failed. You can still use MATCH_METHOD=sift."
    echo "[roma] See requirements-roma.txt for manual steps."
  }
elif [[ "$INSTALL_ROMA" -eq 1 ]]; then
  echo "[roma] skipped (CPU-only setup). RoMa needs a GPU; use MATCH_METHOD=sift."
fi

# ---- verify -----------------------------------------------------------------
echo ""
echo "==============================================================="
echo " Verifying install"
echo "==============================================================="
$PYTHON - << 'PYEOF'
import importlib
mods = ["numpy", "cv2", "skimage", "tifffile", "scipy"]
for m in mods:
    try:
        importlib.import_module(m)
        print(f"  ok   {m}")
    except Exception as e:
        print(f"  FAIL {m}: {e}")

try:
    import torch
    print(f"  ok   torch {torch.__version__}")
    print(f"       CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"       GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"  FAIL torch: {e}")

try:
    import romav2  # noqa
    print("  ok   romav2 (roma/hybrid matchers available)")
except Exception:
    print("  --   romav2 not installed (SIFT only; that's fine)")
PYEOF

echo ""
echo "Done. Quick test:"
echo "  MATCH_METHOD=sift python run_all.py        # CPU, no GPU needed"
echo "  MATCH_METHOD=hybrid python run_all.py       # uses GPU if available"
