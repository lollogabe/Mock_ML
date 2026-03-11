#!/usr/bin/env python3
"""
Colab Setup Helper — Configure environment for Google Colab execution.

This script handles all setup tasks in Colab that would be done via bash
scripts on local/HPC environments. It's safe to run multiple times.

Usage (in Colab cell):
    !python colab_setup.py --setup
    !python colab_setup.py --verify-only
    !python colab_setup.py --full

Features:
- Detects CUDA and installs appropriate PyTorch
- Installs project dependencies
- Verifies installation
- Works around Colab sandbox restrictions
- Provides clear error messages
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description="", check=True):
    """Execute shell command with error handling."""
    if description:
        print(f"==> {description}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
        if check and result.returncode != 0:
            print(f"❌ Error: {description} failed with code {result.returncode}")
            return False
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running '{cmd}': {e}")
        return False


def detect_cuda():
    """Detect CUDA version in Colab environment."""
    print("==> Detecting CUDA version...")

    # In Colab, torch may already have CUDA available
    try:
        import torch

        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            if cuda_version:
                if cuda_version.startswith("12"):
                    print(f"  CUDA {cuda_version} detected → cu121")
                    return "cu121"
                elif cuda_version.startswith("11"):
                    print(f"  CUDA {cuda_version} detected → cu118")
                    return "cu118"
            # If torch is available but version unclear, try to infer
            print("  CUDA available but version unclear, using cu121 (safe default)")
            return "cu121"
        else:
            print("  No CUDA detected, using CPU wheels")
            return "cpu"
    except ImportError:
        print("  PyTorch not yet installed, defaulting to cu121")
        return "cu121"


def install_pytorch(cuda_wheel):
    """Install PyTorch with appropriate CUDA support."""
    print(f"==> Installing PyTorch for {cuda_wheel}")

    if cuda_wheel == "cpu":
        cmd = 'pip install -q "torch>=2.0.0" "torchvision>=0.15.0"'
    else:
        cmd = f'pip install -q "torch>=2.0.0" "torchvision>=0.15.0" --index-url "https://download.pytorch.org/whl/{cuda_wheel}"'

    return run_command(cmd, f"Installing PyTorch ({cuda_wheel})", check=True)


def install_requirements():
    """Install project dependencies."""
    # Check if requirements.txt exists
    if not Path("requirements.txt").exists():
        print("⚠️  requirements.txt not found, skipping dependency installation")
        return True

    # Skip torch/torchvision (already installed by install_pytorch)
    print("==> Installing project requirements (excluding torch/torchvision)")

    # Read requirements and filter out torch-related packages
    try:
        with open("requirements.txt", "r") as f:
            lines = f.readlines()

        filtered_lines = [
            line
            for line in lines
            if not line.strip().startswith("#")
            and not any(
                pkg in line.lower()
                for pkg in ["torch", "torchvision", "pytorch", "cuda"]
            )
        ]

        # Install filtered requirements
        if filtered_lines:
            cmd = f"pip install -q {' '.join([line.strip() for line in filtered_lines if line.strip()])}"
            return run_command(cmd, "Installing dependencies", check=False)  # Don't strict-check
        else:
            print("  No additional dependencies to install")
            return True

    except Exception as e:
        print(f"⚠️  Error reading requirements.txt: {e}")
        return False


def verify_installation():
    """Verify all core packages are installed and importable."""
    print("==> Verifying installation")

    code = """
import sys
try:
    import torch
    import numpy
    import sklearn
    import yaml
    import matplotlib

    print(f"  torch      : {torch.__version__}  (CUDA: {torch.cuda.is_available()})")
    print(f"  numpy      : {numpy.__version__}")
    print(f"  scikit-learn: {sklearn.__version__}")
    print(f"  matplotlib : {matplotlib.__version__}")
    print(f"  yaml       : {yaml.__version__}")

    # Check GPU detailed info if available
    if torch.cuda.is_available():
        print(f"  GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

except ImportError as e:
    print(f"  ❌ ERROR: Missing package: {e}", file=sys.stderr)
    sys.exit(1)
"""

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    print(result.stdout, end="")
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        return False
    return True


def setup_git_ssh(ssh_key_secret=None):
    """
    Setup Git SSH authentication if secret is provided.

    In Colab, you can add SSH key via Secrets and this will configure it.
    Falls back gracefully if SSH key is corrupted or unavailable.
    """
    if ssh_key_secret is None:
        print("⚠️  SSH key not provided, skipping Git setup")
        print("   To use SSH in Colab: add GITHUB_SSH_KEY to Secrets (🔑)")
        print("   OR: use HTTPS clone instead (simpler, no setup needed)")
        return True

    print("==> Configuring Git SSH authentication")

    try:
        ssh_dir = Path.home() / ".ssh"
        ssh_dir.mkdir(exist_ok=True, mode=0o700)

        # Write SSH key
        ssh_key_path = ssh_dir / "id_ed25519"
        ssh_key_path.write_text(ssh_key_secret)
        ssh_key_path.chmod(0o600)

        # Verify key format (should start with -----BEGIN PRIVATE KEY-----)
        key_content = ssh_key_path.read_text()
        if "BEGIN" not in key_content or "PRIVATE" not in key_content:
            print("  ⚠️  Warning: SSH key format looks incorrect")
            print("     (Secrets may have corrupted line breaks)")
            print("     Try: Use HTTPS clone instead, or regenerate key in Colab")
            ssh_key_path.unlink()  # Delete corrupted key
            return False

        # Add GitHub to known_hosts
        cmd = "ssh-keyscan -H github.com >> ~/.ssh/known_hosts 2>/dev/null"
        subprocess.run(cmd, shell=True, capture_output=True)

        # Test SSH connection
        result = subprocess.run(
            "ssh -T git@github.com 2>&1",
            shell=True,
            capture_output=True,
            text=True,
            timeout=5
        )

        if "successfully authenticated" in result.stderr.lower():
            print("  ✓ SSH authentication successful")
            return True
        elif "permission denied" in result.stderr.lower():
            print("  ❌ SSH key rejected by GitHub")
            print("     Check that key is added to: https://github.com/settings/keys")
            ssh_key_path.unlink()
            return False
        else:
            # Inconclusive but likely working
            print("  ⚠️  SSH test inconclusive, but should work")
            print(f"     Full output: {result.stderr}")
            return True

    except Exception as e:
        print(f"  ❌ Error setting up SSH: {e}")
        print("     Fallback: Use HTTPS clone (https://github.com/USERNAME/REPO.git)")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Setup Google Colab environment for Mock_ML project"
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Run full setup (detect, install, verify)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing installation",
    )
    parser.add_argument(
        "--full", action="store_true", help="Alias for --setup (comprehensive setup)"
    )
    parser.add_argument(
        "--with-ssh",
        action="store_true",
        help="Setup SSH auth (requires GITHUB_SSH_KEY secret in Colab)",
    )

    args = parser.parse_args()

    # Default to --setup if no args
    if not any([args.setup, args.verify_only, args.full]):
        args.setup = True

    print("═" * 70)
    print("  Google Colab Setup for Mock_ML Project")
    print("═" * 70)

    # Step 1: Detect CUDA
    cuda = detect_cuda()

    # Step 2: Install PyTorch
    if args.setup or args.full:
        if not install_pytorch(cuda):
            print("❌ PyTorch installation failed")
            return 1

        # Step 3: Install remaining dependencies
        if not install_requirements():
            print("⚠️  Some dependencies failed to install (continuing anyway)")

        # Step 4: Setup SSH if requested
        if args.with_ssh:
            try:
                from google.colab import userdata

                try:
                    ssh_key = userdata.get("GITHUB_SSH_KEY")
                    setup_git_ssh(ssh_key)
                except Exception as e:
                    print(f"⚠️  Could not retrieve SSH key from Secrets: {e}")
            except ImportError:
                print("⚠️  google.colab not available (not running in Colab?)")

    # Step 5: Verify installation
    if not verify_installation():
        print("❌ Installation verification failed")
        return 1

    print("")
    print("╔" + "═" * 68 + "╗")
    print("║  ✓  Colab setup complete!                                            ║")
    print("╠" + "═" * 68 + "╣")
    print("║  Next steps:                                                          ║")
    print("║    1. Clone repository:                                               ║")
    print("║       !git clone git@github.com:YOUR_USERNAME/Mock_ML.git             ║")
    print("║       %cd Mock_ML                                                     ║")
    print("║                                                                       ║")
    print("║    2. Download data:                                                  ║")
    print("║       !python scripts/preprocess.py --group 37                        ║")
    print("║                                                                       ║")
    print("║    3. Start training:                                                 ║")
    print("║       !python scripts/train.py --config configs/config.yaml --device cuda ║")
    print("║                                                                       ║")
    print("║    4. Evaluate:                                                       ║")
    print("║       !python scripts/evaluate.py --checkpoint checkpoints/ae_best.pt ║")
    print("╚" + "═" * 68 + "╝")

    return 0


if __name__ == "__main__":
    sys.exit(main())
