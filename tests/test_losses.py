"""
test_losses.py — Unit tests for hybrid BCE + Dice loss function.

Tests:
  1. Perfect prediction → Dice loss ≈ 0, hybrid loss ≈ 0.
  2. Worst prediction (all zeros against all-one target) → Dice = 1.
  3. Loss is a scalar tensor.
  4. Gradients flow through the loss.
  5. All-background prediction (no tumor) handled correctly (empty Dice).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch

from spiking_useg.training.losses import hybrid_loss, dice_loss


class TestDiceLoss:
    """Tests for the soft Dice loss component."""

    def test_perfect_prediction_is_zero(self):
        """Dice loss = 0 when pred == target."""
        pred = torch.ones(2, 3, 8, 8)
        target = torch.ones(2, 3, 8, 8)
        loss = dice_loss(pred, target)
        assert loss.item() < 1e-4, f"Expected ~0, got {loss.item():.6f}"

    def test_worst_prediction(self):
        """Dice loss ≈ 1 when pred is all zeros and target is all ones."""
        pred = torch.zeros(2, 3, 8, 8)
        target = torch.ones(2, 3, 8, 8)
        loss = dice_loss(pred, target)
        # With numerator=0 and denom=N, dice_coeff=eps/(N+eps)≈0, dice_loss≈1
        assert loss.item() > 0.99, f"Expected ~1, got {loss.item():.6f}"

    def test_all_background_no_nan(self):
        """Dice loss should not be NaN when both pred and target are all zeros."""
        pred = torch.zeros(2, 3, 8, 8)
        target = torch.zeros(2, 3, 8, 8)
        loss = dice_loss(pred, target)
        assert not torch.isnan(loss), "NaN in Dice loss for all-zero inputs"


class TestHybridLoss:
    """Tests for the full hybrid BCE + Dice loss."""

    def test_returns_scalar(self):
        """hybrid_loss should return a scalar tensor."""
        pred = torch.sigmoid(torch.randn(1, 3, 16, 16))
        target = (torch.rand(1, 3, 16, 16) > 0.7).float()
        loss = hybrid_loss(pred, target)
        assert loss.dim() == 0, "Loss should be a scalar (0-dim tensor)"

    def test_gradients_flow(self):
        """Gradients should flow through hybrid_loss."""
        x = torch.randn(1, 3, 8, 8, requires_grad=True)
        pred = torch.sigmoid(x)  # pred is non-leaf; check grad on x (the leaf)
        target = (torch.rand(1, 3, 8, 8) > 0.5).float()
        loss = hybrid_loss(pred, target)
        loss.backward()
        assert x.grad is not None, "No gradient computed"
        assert not torch.isnan(x.grad).any(), "NaN in gradients"

    def test_perfect_prediction_low_loss(self):
        """Nearly perfect prediction should yield a very low loss."""
        pred = torch.full((1, 3, 8, 8), 0.999)   # near 1 everywhere
        target = torch.ones(1, 3, 8, 8)
        loss = hybrid_loss(pred, target)
        assert loss.item() < 0.05, f"Expected low loss for near-perfect pred, got {loss.item():.4f}"

    def test_loss_non_negative(self):
        """Loss must be non-negative."""
        pred = torch.sigmoid(torch.randn(2, 3, 16, 16))
        target = (torch.rand(2, 3, 16, 16) > 0.5).float()
        loss = hybrid_loss(pred, target)
        assert loss.item() >= 0, f"Negative loss: {loss.item()}"
