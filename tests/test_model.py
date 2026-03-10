"""
tests/test_model.py — Unit tests for src/model.py
"""

import pytest
import torch

from src.model import AE, Decoder, Encoder, build_model


@pytest.fixture
def ae():
    return build_model(hidden_channels=8, latent_dim=4)


class TestEncoder:
    def test_output_shape(self):
        enc = Encoder(hidden_channels=8, latent_dim=4)
        x = torch.zeros(2, 1, 100, 100)
        z = enc(x)
        assert z.shape == (2, 4), f"Expected (2,4), got {z.shape}"

    def test_batch_size_invariant(self):
        enc = Encoder(hidden_channels=8, latent_dim=4)
        for bs in (1, 4, 16):
            z = enc(torch.zeros(bs, 1, 100, 100))
            assert z.shape == (bs, 4)


class TestDecoder:
    def test_output_shape(self):
        dec = Decoder(hidden_channels=8, latent_dim=4)
        z = torch.zeros(2, 4)
        out = dec(z)
        assert out.shape == (2, 1, 100, 100), f"Expected (2,1,100,100), got {out.shape}"

    def test_output_range(self):
        dec = Decoder(hidden_channels=8, latent_dim=4)
        z = torch.randn(4, 4)
        out = dec(z)
        assert out.min() >= 0.0 - 1e-6
        assert out.max() <= 1.0 + 1e-6


class TestAE:
    def test_reconstruction_shape(self, ae):
        x = torch.rand(3, 1, 100, 100)
        xhat = ae(x)
        assert xhat.shape == x.shape

    def test_gradient_flows(self, ae):
        x = torch.rand(2, 1, 100, 100)
        xhat = ae(x)
        loss = torch.nn.BCELoss()(xhat, x)
        loss.backward()
        # Check at least one parameter has a gradient
        has_grad = any(p.grad is not None for p in ae.parameters())
        assert has_grad

    def test_build_model_defaults(self):
        m = build_model()
        assert isinstance(m, AE)
        assert m.encoder.latent_dim == 4
        assert m.encoder.hidden_channels == 32
