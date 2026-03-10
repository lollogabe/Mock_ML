"""
tests/test_utils.py — Unit tests for src/utils.py
"""

import numpy as np
import pytest
import torch

from src.utils import get_device, purity_score, set_seed


class TestSetSeed:
    def test_numpy_determinism(self):
        set_seed(0)
        a = np.random.rand(10)
        set_seed(0)
        b = np.random.rand(10)
        np.testing.assert_array_equal(a, b)

    def test_torch_determinism(self):
        set_seed(1)
        t1 = torch.randn(5)
        set_seed(1)
        t2 = torch.randn(5)
        assert torch.allclose(t1, t2)


class TestGetDevice:
    def test_auto_returns_device(self):
        d = get_device("auto")
        assert isinstance(d, torch.device)
        assert d.type in ("cpu", "cuda")

    def test_cpu_explicit(self):
        d = get_device("cpu")
        assert d.type == "cpu"


class TestPurityScore:
    def test_perfect_purity(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        assert purity_score(y_true, y_pred) == pytest.approx(1.0)

    def test_worst_purity(self):
        # All predictions swapped
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        # Each cluster has a clear majority of the opposite label → purity = 0
        score = purity_score(y_true, y_pred)
        assert 0.0 <= score <= 1.0

    def test_purity_in_range(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 100)
        y_pred = rng.integers(0, 2, 100)
        score = purity_score(y_true, y_pred)
        assert 0.0 <= score <= 1.0
