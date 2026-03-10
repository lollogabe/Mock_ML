"""
scripts/preprocess.py — Download and verify the CERN jet-image datasets.

Usage:
    python scripts/preprocess.py --group 37 --data-dir data/raw
"""

import argparse
import logging
import sys
import os

# Allow running from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import setup_logging
from src.data_loader import download_data, load_tensors


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download CERN jet-image datasets.")
    p.add_argument("--group",    type=int,   default=37,       help="Dataset group identifier (default: 37)")
    p.add_argument("--data-dir", type=str,   default="data/raw", help="Directory to save .npz files")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(log_dir="logs", log_file="preprocess.log")
    logger = logging.getLogger(__name__)

    logger.info(f"Downloading data for group {args.group} → {args.data_dir}")
    download_data(group=args.group, data_dir=args.data_dir)

    logger.info("Verifying downloaded files …")
    normal_t, low_t, high_t = load_tensors(data_dir=args.data_dir)

    assert normal_t.shape == (12000, 1, 100, 100), f"Unexpected shape: {normal_t.shape}"
    assert low_t.shape[1:] == (1, 100, 100),  f"Unexpected shape: {low_t.shape}"
    assert high_t.shape[1:] == (1, 100, 100), f"Unexpected shape: {high_t.shape}"

    logger.info("✓ All datasets downloaded and verified.")
    print(f"normal: {tuple(normal_t.shape)}")
    print(f"low:    {tuple(low_t.shape)}")
    print(f"high:   {tuple(high_t.shape)}")


if __name__ == "__main__":
    main()
