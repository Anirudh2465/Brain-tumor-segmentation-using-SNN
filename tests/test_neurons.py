"""
test_neurons.py — Unit tests for PLIF neuron dynamics.

Tests:
  1. Membrane update equation matches hand-computed values.
  2. Spike fires when u > θ, not when u ≤ θ.
  3. reset_state() zeroes out the membrane potential.
  4. Surrogate gradient has the correct arctan form.
  5. Decay λ is always in (0, 1) regardless of τ_param value.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch
import torch.nn as nn

from spiking_useg.models.neurons import PLIFLayer
from spiking_useg.models.surrogate import heaviside_arctan


class TestSurrogateGradient:
    """Tests for the arctan surrogate gradient."""

    def test_forward_is_heaviside(self):
        """Heaviside: 0 for x<=0, 1 for x>0."""
        x = torch.tensor([-1.0, -0.01, 0.0, 0.01, 1.0])
        out = heaviside_arctan(x)
        expected = torch.tensor([0., 0., 0., 1., 1.])
        assert torch.all(out == expected), f"Expected {expected}, got {out}"

    def test_backward_is_arctan(self):
        """d/dx surrogate at x=0 should be 1/(1+(π·0)²) = 1."""
        x = torch.tensor([0.0], requires_grad=True)
        out = heaviside_arctan(x)
        out.backward()
        assert abs(x.grad.item() - 1.0) < 1e-5, f"Expected grad≈1.0 at x=0, got {x.grad.item()}"

    def test_backward_at_nonzero(self):
        """d/dx surrogate at x=1 should be 1/(1+π²) ≈ 0.0920."""
        x = torch.tensor([1.0], requires_grad=True)
        out = heaviside_arctan(x)
        out.backward()
        expected_grad = 1.0 / (1.0 + (torch.pi * 1.0) ** 2)
        assert abs(x.grad.item() - expected_grad) < 1e-5, (
            f"Expected grad≈{expected_grad:.4f} at x=1, got {x.grad.item():.4f}"
        )


class TestPLIFLayer:
    """Tests for PLIF membrane dynamics."""

    def test_decay_range(self):
        """λ = sigmoid(τ) must always be in (0, 1)."""
        neuron = PLIFLayer(num_channels=4)
        for tau_val in [-10.0, -1.0, 0.0, 1.0, 10.0]:
            neuron.tau_param.data.fill_(tau_val)
            lam = neuron.decay.item()
            assert 0.0 < lam < 1.0, f"λ={lam} out of (0,1) for τ_param={tau_val}"

    def test_spike_threshold(self):
        """Neuron should spike only when membrane potential exceeds threshold."""
        neuron = PLIFLayer(num_channels=1, threshold=1.0)
        # Large input → spike
        large_input = torch.full((1, 1, 2, 2), 2.0)
        s = neuron(large_input)
        assert s.sum().item() > 0, "Expected spikes for large input"

        neuron.reset_state()
        # Small input → no spike
        small_input = torch.full((1, 1, 2, 2), 0.1)
        s = neuron(small_input)
        # After one step with u = 0.1 < 1.0, should be all zeros
        assert s.sum().item() == 0, "Expected no spikes for small input"

    def test_reset_state_clears_membrane(self):
        """reset_state() must zero out membrane potential and previous spike."""
        neuron = PLIFLayer(num_channels=1, threshold=1.0)
        x = torch.full((1, 1, 4, 4), 2.0)
        neuron(x)  # First step — builds up membrane potential
        assert neuron._u is not None, "State should be set after forward"
        neuron.reset_state()
        assert neuron._u is None, "State should be None after reset_state()"

    def test_membrane_update_formula(self):
        """Manually verify u_t = λ·u_{t-1} + I_t − θ·s_{t-1}."""
        neuron = PLIFLayer(num_channels=1, threshold=1.0, tau_init=0.0)  # λ=0.5
        lam = 0.5
        theta = 1.0

        # Step 1: u_0 = 0 → I_1 = 0.3 → u_1 = 0.5×0 + 0.3 − 1.0×0 = 0.3, no spike
        I1 = torch.tensor([[[[0.3]]]])
        s1 = neuron(I1)
        assert s1.item() == 0.0
        assert abs(neuron._u.item() - 0.3) < 1e-5, f"u after step 1: {neuron._u.item()}"

        # Step 2: I_2 = 1.5 → u_2 = 0.5×0.3 + 1.5 − 0 = 1.65 → spike
        I2 = torch.tensor([[[[1.5]]]])
        s2 = neuron(I2)
        assert s2.item() == 1.0, "Expected spike at step 2"

    def test_no_nan_in_output(self):
        """No NaN or Inf should appear in neuron output."""
        neuron = PLIFLayer(num_channels=8)
        x = torch.randn(2, 8, 16, 16)
        for _ in range(5):
            out = neuron(x)
            assert not torch.isnan(out).any(), "NaN in neuron output"
            assert not torch.isinf(out).any(), "Inf in neuron output"
