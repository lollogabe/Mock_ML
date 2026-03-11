#!/usr/bin/env bash
# =============================================================================
# scripts/utils.sh — Shared shell utilities for environment setup
#
# Provides reusable functions for both local (setup.sh) and HPC (setup_hpc.sh)
# environments to reduce code duplication.
#
# Source this file in other setup scripts:
#   source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"
# =============================================================================

set -euo pipefail

# ── CUDA Detection ─────────────────────────────────────────────────────────
# Returns: "cu121", "cu118", or "cpu"

detect_cuda_via_nvidia_smi() {
    # Detect CUDA version using nvidia-smi command.
    # Falls back to 'cpu' if not available.
    if ! command -v nvidia-smi &>/dev/null; then
        echo "cpu"
        return
    fi

    local ver
    # Use portable grep (compatible with macOS and Linux)
    ver=$(nvidia-smi 2>/dev/null | grep "CUDA Version:" | sed -E 's/.*CUDA Version: ([0-9]+\.[0-9]+).*/\1/' || echo "")

    if [[ -z "$ver" ]]; then
        echo "cpu"
    elif [[ "$ver" == 12.* ]]; then
        echo "cu121"
    elif [[ "$ver" == 11.* ]]; then
        echo "cu118"
    else
        echo "cpu"
    fi
}

detect_cuda_via_modules() {
    # Detect CUDA version using loaded modules (HPC clusters).
    # Used as fallback when nvidia-smi is unavailable.
    if command -v module &>/dev/null; then
        if module list 2>&1 | grep -q "cuda/12"; then
            echo "cu121"
        elif module list 2>&1 | grep -q "cuda/11"; then
            echo "cu118"
        else
            echo "cpu"
        fi
    else
        echo "cpu"
    fi
}

detect_cuda() {
    # Unified CUDA detection: tries nvidia-smi first, then modules, then defaults to cpu.
    # Environment override:
    #     CUDA=cu121 detect_cuda     # forces cu121
    #     CUDA=cpu   detect_cuda     # forces cpu
    # Allow explicit override via CUDA env var
    if [[ -n "${CUDA:-}" ]]; then
        echo "$CUDA"
        return
    fi
    
    # Try nvidia-smi (usually available)
    local cuda_from_smi
    cuda_from_smi=$(detect_cuda_via_nvidia_smi)
    if [[ "$cuda_from_smi" != "cpu" ]]; then
        echo "$cuda_from_smi"
        return
    fi
    
    # Fallback to module inspection (HPC)
    detect_cuda_via_modules
}

# ── Python Version Detection ───────────────────────────────────────────────

verify_python() {
    # Verify Python executable is available and output version.
    # Prefers $PYTHON env var if set; otherwise uses system python3/python.
    local python_exe="${PYTHON:-python3}"
    
    # Try PYTHON env var first
    if [[ -n "${PYTHON:-}" ]]; then
        python_exe="$PYTHON"
    elif command -v python3 &>/dev/null; then
        python_exe="python3"
    elif command -v python &>/dev/null; then
        python_exe="python"
    else
        echo "ERROR: Python not found in PATH" >&2
        return 1
    fi
    
    "$python_exe" --version
    echo "$python_exe"
}

# ── Virtual Environment Setup ──────────────────────────────────────────────

create_venv() {
    # Create a Python virtual environment.
    # Usage:
    #     create_venv /path/to/venv          # basic setup
    #     create_venv /path/to/venv --system-site-packages  # inherit system packages
    local venv_path="$1"
    local venv_flags=("${@:2}")  # all args after first

    if [[ -d "$venv_path" ]]; then
        echo "==> Removing existing venv at $venv_path"
        rm -rf "$venv_path"
    fi

    echo "==> Creating virtual environment at $venv_path"
    # Use detected Python executable or fall back to python3
    local python_exe="${PYTHON:-python3}"
    "$python_exe" -m venv "${venv_flags[@]}" "$venv_path"
}

activate_venv() {
    # Activate a virtual environment by sourcing its activate script.
    # Usage:
    #     source <(activate_venv /path/to/venv)
    local venv_path="$1"
    
    if [[ ! -f "$venv_path/bin/activate" ]]; then
        echo "ERROR: venv not found at $venv_path/bin/activate" >&2
        return 1
    fi
    
    # shellcheck disable=SC1091
    source "$venv_path/bin/activate"
}

# ── Pip Installation ──────────────────────────────────────────────────────

install_pytorch() {
    # Install PyTorch with CUDA-specific wheel.
    # Usage:
    #     install_pytorch cu121
    #     install_pytorch cpu
    local cuda_wheel="${1:-cu121}"
    
    echo "==> Installing PyTorch for ${cuda_wheel}"
    
    if [[ "$cuda_wheel" == "cpu" ]]; then
        pip install --quiet "torch>=2.0.0" "torchvision>=0.15.0"
    else
        pip install --quiet \
            "torch>=2.0.0" "torchvision>=0.15.0" \
            --index-url "https://download.pytorch.org/whl/${cuda_wheel}"
    fi
}

install_requirements() {
    # Install dependencies from requirements.txt.
    # Optionally filters out specified packages.
    # Usage:
    #     install_requirements                           # all packages
    #     install_requirements torch torchvision        # skip these packages
    local skip_packages=("${@:-}")

    if [[ ${#skip_packages[@]} -eq 0 ]]; then
        pip install --quiet -r requirements.txt
    else
        # Build grep pattern to exclude packages (e.g., "^torch$|^torchvision$")
        local pattern=""
        for pkg in "${skip_packages[@]}"; do
            if [[ -z "$pattern" ]]; then
                pattern="^${pkg}[>=<~!]|^${pkg}$"
            else
                pattern="${pattern}|^${pkg}[>=<~!]|^${pkg}$"
            fi
        done

        grep -v -E "$pattern" requirements.txt | pip install --quiet -r /dev/stdin
    fi
}

# ── Verification ──────────────────────────────────────────────────────────

verify_installation() {
    # Verify core dependencies are installed and importable.
    echo "==> Verifying installation"
    python - <<'EOF'
import sys
try:
    import torch
    import numpy
    import sklearn
    import yaml
    print(f"  torch      : {torch.__version__}  (CUDA: {torch.cuda.is_available()})")
    print(f"  numpy      : {numpy.__version__}")
    print(f"  scikit-learn: {sklearn.__version__}")
except ImportError as e:
    print(f"  ERROR: {e}", file=sys.stderr)
    sys.exit(1)
EOF
}

# Export functions so they're available to scripts that source this file
export -f detect_cuda_via_nvidia_smi detect_cuda_via_modules detect_cuda
export -f verify_python create_venv activate_venv
export -f install_pytorch install_requirements verify_installation
