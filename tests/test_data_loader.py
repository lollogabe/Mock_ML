"""
tests/test_data_loader.py — Unit tests for src/data_loader.py

Uses in-memory mock .npz files so no network access or real data is required.
"""

import os
import tempfile

import numpy as np
import pytest
import torch

from src.data_loader import build_dataloaders, load_tensors


@pytest.fixture
def mock_data_dir():
    """Create a temporary directory with tiny mock .npz files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Normal data: 100 samples of 100×100
        np.savez(
            os.path.join(tmpdir, "Normal_data.npz"),
            normal_data=np.random.rand(100, 100, 100).astype(np.float32),
        )
        # Low / high anomaly data: 20 samples of 100×100
        np.savez(
            os.path.join(tmpdir, "Test_data_low.npz"),
            test_data=np.random.rand(20, 100, 100).astype(np.float32),
        )
        np.savez(
            os.path.join(tmpdir, "Test_data_high.npz"),
            test_data=np.random.rand(20, 100, 100).astype(np.float32),
        )
        yield tmpdir


class TestLoadTensors:
    def test_shapes(self, mock_data_dir):
        normal_t, low_t, high_t = load_tensors(data_dir=mock_data_dir)
        assert normal_t.shape == (100, 1, 100, 100)
        assert low_t.shape    == (20,  1, 100, 100)
        assert high_t.shape   == (20,  1, 100, 100)

    def test_dtype(self, mock_data_dir):
        normal_t, _, _ = load_tensors(data_dir=mock_data_dir)
        assert normal_t.dtype == torch.float32


class TestBuildDataloaders:
    def test_four_dataloaders_returned(self, mock_data_dir):
        normal_t, low_t, high_t = load_tensors(data_dir=mock_data_dir)
        dls = build_dataloaders(normal_t, low_t, high_t,
                                batch_size=8, test_frac=0.2, seed=42)
        assert len(dls) == 4

    def test_train_batch_size(self, mock_data_dir):
        normal_t, low_t, high_t = load_tensors(data_dir=mock_data_dir)
        dl_train, _, _, _ = build_dataloaders(normal_t, low_t, high_t,
                                              batch_size=8, test_frac=0.2, seed=42)
        xb, yb = next(iter(dl_train))
        assert xb.shape[0] == 8       # batch size
        assert xb.shape[1:] == (1, 100, 100)

    def test_low_high_batch_size_1(self, mock_data_dir):
        normal_t, low_t, high_t = load_tensors(data_dir=mock_data_dir)
        _, _, dl_low, dl_high = build_dataloaders(normal_t, low_t, high_t,
                                                  batch_size=8, test_frac=0.2, seed=42)
        xb, _ = next(iter(dl_low))
        assert xb.shape[0] == 1

    def test_train_test_split_sizes(self, mock_data_dir):
        normal_t, low_t, high_t = load_tensors(data_dir=mock_data_dir)
        dl_train, dl_n_test, _, _ = build_dataloaders(
            normal_t, low_t, high_t, batch_size=4, test_frac=0.2, seed=42
        )
        n_train_samples = sum(xb.shape[0] for xb, _ in dl_train)
        n_test_samples  = sum(xb.shape[0] for xb, _ in dl_n_test)
        # 80 train (drop_last may drop 0), 20 test
        assert n_test_samples == 20
        assert n_train_samples <= 80
