#!/usr/bin/env bash
# =============================================================================
# setup.sh — Local virtual environment setup (macOS / Linux workstation)
#
# Usage:
#   bash setup.sh               # auto-detect CUDA
#   CUDA=cu121 bash setup.sh    # force CUDA 12.1 wheel
#   CUDA=cpu   bash setup.sh    # force CPU-only
#
# For HPC / CINECA use setup_hpc.sh instead.
# =============================================================================
set -euo pipefail

PYTHON=${PYTHON:-python3}
VENV_DIR=${VENV_DIR:-"venv"}

# ── Detect CUDA version for PyTorch wheel selection ───────────────────────────
detect_cuda() {
    if ! command -v nvidia-smi &>/dev/null; then
        echo "cpu"
        return
    fi
    local ver
    ver=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" 2>/dev/null || echo "")
    if [[ -z "$ver" ]]; then
        echo "cpu"
    elif [[ "$ver" == 12.* ]]; then
        echo "cu121"
    elif [[ "$ver" == 11.8* ]]; then
        echo "cu118"
    elif [[ "$ver" == 11.* ]]; then
        echo "cu118"        # closest available wheel
    else
        echo "cpu"
    fi
}

CUDA=${CUDA:-$(detect_cuda)}
echo "==> CUDA target: ${CUDA}"

# ── Create virtual environment ─────────────────────────────────────────────────
# ── Create virtual environment ─────────────────────────────────────────────────
if [[ -d "$VENV_DIR" ]]; then
    echo "==> Cleaning up existing venv..."
    rm -rf "$VENV_DIR"
fi

echo "==> Creating virtual environment in ./${VENV_DIR}/"
"$PYTHON" -m venv "$VENV_DIR"
# --copies: avoids symlink fragility; important when venv is moved or shared

echo "==> Activating environment"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "==> Upgrading pip / wheel / setuptools"
pip install --quiet --upgrade pip wheel setuptools

# ── PyTorch (target-specific wheel) ───────────────────────────────────────────
if [[ "$CUDA" == "cpu" ]]; then
    echo "==> Installing CPU-only PyTorch"
    pip install --quiet "torch>=2.0.0" "torchvision>=0.15.0"
else
    echo "==> Installing PyTorch for ${CUDA}"
    pip install --quiet \
        "torch>=2.0.0" "torchvision>=0.15.0" \
        --index-url "https://download.pytorch.org/whl/${CUDA}"
fi

# ── Remaining dependencies ─────────────────────────────────────────────────────
echo "==> Installing project requirements"
pip install --quiet -r requirements.txt

# ── Verification ──────────────────────────────────────────────────────────────
echo ""
echo "==> Verifying installation"
python - <<'EOF'
import torch, numpy, sklearn, yaml, dotenv, scipy, tqdm
print(f"  torch      : {torch.__version__}  (CUDA available: {torch.cuda.is_available()})")
print(f"  numpy      : {numpy.__version__}")
print(f"  scikit-learn: {sklearn.__version__}")
EOF

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ✓  Setup complete!                                          ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Activate  :  source venv/bin/activate                      ║"
echo "║  Download  :  python scripts/preprocess.py --group 37       ║"
echo "║  Train     :  python scripts/train.py --config configs/config.yaml ║"
echo "║  Evaluate  :  python scripts/evaluate.py --checkpoint checkpoints/ae_best.pt ║"
echo "║  Tests     :  python -m pytest tests/ -v                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
