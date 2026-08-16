"""
test_ensemble.py — Integration tests for SpikingUSegNet and view permutation.

Tests:
  1. Full forward pass (all 3 views) produces correct shapes.
  2. predict_volume correctly permutes output to canonical (3, 160, 192, 152) frame.
  3. model.reset_states() before each new subject (state isolation test).
  4. SpikingUSegNet end-to-end forward pass on synthetic data: no NaN / Inf.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest
import torch

from spiking_useg.models.spiking_unet import SpikingUSegNet
from spiking_useg.inference.predict import predict_volume


CANONICAL_SHAPE = (160, 192, 152)  # H, W, D


class TestSpikingUSegNetForward:
    """Shape and correctness tests for SpikingUSegNet forward pass."""

    @pytest.fixture
    def model(self):
        return SpikingUSegNet()

    def test_output_shape(self, model):
        """Output should be (B, 3, H, W) for a (B, 4, H, W) input."""
        model.reset_states()
        x = torch.randn(1, 4, 64, 64)
        out = model(x)
        assert out.shape == (1, 3, 64, 64), f"Expected (1, 3, 64, 64), got {out.shape}"

    def test_output_in_range(self, model):
        """Output sigmoid should be in [0, 1] (Model outputs logits)."""
        model.reset_states()
        x = torch.randn(1, 4, 32, 32)
        out = model(x).sigmoid()
        assert out.min().item() >= 0.0 and out.max().item() <= 1.0, (
            f"Output out of [0,1]: min={out.min().item()}, max={out.max().item()}"
        )

    def test_no_nan_inf(self, model):
        """No NaN or Inf in the output."""
        model.reset_states()
        for t in range(5):
            x = torch.randn(1, 4, 32, 32)
            out = model(x)
            assert not torch.isnan(out).any(), f"NaN at step {t}"
            assert not torch.isinf(out).any(), f"Inf at step {t}"

    def test_reset_states_isolation(self, model):
        """Two subjects should get independent predictions (state isolated by reset)."""
        model.eval()
        x = torch.ones(1, 4, 32, 32)

        # Subject 1
        model.reset_states()
        out1 = model(x)

        # Subject 2 — with reset, first step should give same result as subject 1
        model.reset_states()
        out2 = model(x)

        assert torch.allclose(out1, out2, atol=1e-5), (
            "Outputs differ after reset_states — state contamination between subjects"
        )

    def test_state_contamination_without_reset(self, model):
        """Without reset, consecutive subjects should give DIFFERENT first-step outputs."""
        model.eval()
        x = torch.ones(1, 4, 32, 32)

        model.reset_states()
        # Run several steps to build up state
        for _ in range(5):
            model(x * 2.0)

        # Without reset, next forward should differ from fresh start
        out_contaminated = model(x)

        model.reset_states()
        out_clean = model(x)

        # They should differ due to accumulated membrane state
        assert not torch.allclose(out_contaminated, out_clean, atol=1e-4), (
            "Expected state contamination without reset_states(), but outputs were identical"
        )


class TestPredictVolume:
    """Tests for predict_volume canonical orientation."""

    @pytest.mark.parametrize("view,expected_shape", [
        ("axial",    (3, 160, 192, 152)),
        ("coronal",  (3, 160, 192, 152)),
        ("sagittal", (3, 160, 192, 152)),
    ])
    def test_output_canonical_shape(self, view, expected_shape):
        """predict_volume should return (3, H, W, D) for all views."""
        model = SpikingUSegNet()
        model.eval()
        device = torch.device("cpu")

        # Synthetic preprocessed volume
        data = np.random.rand(4, *CANONICAL_SHAPE).astype(np.float32)

        result = predict_volume(model, data, view=view, device=device)
        assert result.shape == expected_shape, (
            f"View={view}: expected {expected_shape}, got {result.shape}"
        )

    def test_predict_volume_range(self):
        """Predicted probability volume should be in [0, 1]."""
        model = SpikingUSegNet()
        model.eval()
        data = np.random.rand(4, *CANONICAL_SHAPE).astype(np.float32)
        result = predict_volume(model, data, view="axial", device=torch.device("cpu"))
        assert result.min() >= 0.0 and result.max() <= 1.0, (
            f"Probabilities out of [0,1]: min={result.min():.4f}, max={result.max():.4f}"
        )
