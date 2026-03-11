#!/usr/bin/env bash
# =============================================================================
# setup.sh — Local virtual environment setup (macOS / Linux workstation)
#
# This script sets up a Python virtual environment for development on a local
# workstation (macOS or Linux). It automatically detects your CUDA version and
# installs the appropriate PyTorch wheel.
#
# Usage:
#   bash setup.sh               # auto-detect CUDA version
#   CUDA=cu121 bash setup.sh    # force CUDA 12.1 wheel
#   CUDA=cpu   bash setup.sh    # force CPU-only installation
#
# For HPC cluster environments (CINECA, etc.), use setup_hpc.sh instead.
#
# =============================================================================
set -euo pipefail

# ── Find best compatible Python version ────────────────────────────────────
# numba/llvmlite have issues with Python 3.12 on macOS; prefer 3.11 or 3.10
find_best_python() {
    for py_ver in python3.11 python3.10 python3.9 python3; do
        if command -v "$py_ver" &>/dev/null; then
            echo "$py_ver"
            return 0
        fi
    done
    echo "python3"  # fallback
}

PYTHON=${PYTHON:-$(find_best_python)}

# ── Source shared utilities ────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/scripts/utils.sh"

VENV_DIR=${VENV_DIR:-"venv"}

echo "═══════════════════════════════════════════════════════════════════"
echo "  Local Python Environment Setup"
echo "═══════════════════════════════════════════════════════════════════"

# ── Detect CUDA version ────────────────────────────────────────────────────
CUDA=$(detect_cuda)
echo "==> CUDA target: ${CUDA}"

# ── Create and activate virtual environment ────────────────────────────────
create_venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "==> Upgrading pip / wheel / setuptools"
pip install --quiet --upgrade pip wheel setuptools

# ── Install PyTorch ───────────────────────────────────────────────────────
install_pytorch "$CUDA"

# ── Install remaining dependencies ─────────────────────────────────────────
echo "==> Installing project requirements"
pip install --quiet -r requirements.txt

# ── Verify installation ────────────────────────────────────────────────────
verify_installation

echo ""
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║  ✓  Setup complete!                                               ║"
echo "╠═══════════════════════════════════════════════════════════════════╣"
echo "║  Activate environment:                                            ║"
echo "║    source ${VENV_DIR}/bin/activate                               ║"
echo "║                                                                   ║"
echo "║  Next steps:                                                      ║"
echo "║    1. Download data:                                              ║"
echo "║       python scripts/preprocess.py --group 37                     ║"
echo "║                                                                   ║"
echo "║    2. Run tests (optional):                                       ║"
echo "║       python -m pytest tests/ -v                                  ║"
echo "║                                                                   ║"
echo "║    3. Start training:                                             ║"
echo "║       python scripts/train.py --config configs/config.yaml        ║"
echo "║                                                                   ║"
echo "║    4. Evaluate (after training):                                  ║"
echo "║       python scripts/evaluate.py --checkpoint checkpoints/ae_best.pt ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
