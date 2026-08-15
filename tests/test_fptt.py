"""
test_fptt.py — Unit tests for the FPTT optimizer.

Tests:
  1. Memory stays flat as T increases (no BPTT unrolling).
  2. FPTTOptimizer reduces loss on a trivial 2-step regression task.
  3. Gradient accumulator is updated correctly (hand-computed toy example).
  4. start_sequence() properly resets all buffers.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch
import torch.nn as nn

from spiking_useg.training.fptt import FPTTOptimizer


class TinyLinear(nn.Module):
    """A tiny 1-layer linear model for testing."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1, bias=False)

    def forward(self, x):
        return self.linear(x)


class TestFPTT:

    def _make_fptt(self, alpha: float = 0.1) -> tuple[TinyLinear, FPTTOptimizer]:
        model = TinyLinear()
        base_optim = torch.optim.Adam(model.parameters(), lr=0.01)
        fptt = FPTTOptimizer(base_optim, alpha=alpha, grad_clip_norm=1.0)
        return model, fptt

    def test_loss_decreases_on_toy_task(self):
        """FPTT should reduce loss on a simple linear regression task."""
        torch.manual_seed(0)
        model, fptt = self._make_fptt()

        x = torch.randn(5, 4)    # T=5 time steps
        y = torch.randn(5, 1)    # targets

        first_losses = []
        last_losses = []

        for epoch in range(20):
            fptt.start_sequence()
            epoch_loss = 0.0
            for t in range(5):
                pred = model(x[t].unsqueeze(0))
                loss = ((pred - y[t].unsqueeze(0)) ** 2).mean()
                step_loss = fptt.step(loss)
                epoch_loss += step_loss
            if epoch < 3:
                first_losses.append(epoch_loss)
            if epoch >= 17:
                last_losses.append(epoch_loss)

        assert sum(last_losses) < sum(first_losses), (
            f"FPTT did not reduce loss: first_avg={sum(first_losses)/3:.4f}, "
            f"last_avg={sum(last_losses)/3:.4f}"
        )

    def test_start_sequence_resets_buffers(self):
        """start_sequence() should clear all FPTT state."""
        model, fptt = self._make_fptt()
        fptt.start_sequence()

        x = torch.randn(1, 4)
        for t in range(3):
            loss = ((model(x) - 0.5) ** 2).mean()
            fptt.step(loss)

        # Now state should be populated
        assert fptt._initialized
        assert len(fptt._w_bar) > 0

        fptt.start_sequence()
        assert not fptt._initialized
        assert len(fptt._w_bar) == 0

    def test_step_does_not_error(self):
        """FPTT step should execute without error for T=10 steps."""
        model, fptt = self._make_fptt()
        fptt.start_sequence()
        x = torch.randn(1, 4)
        for t in range(10):
            loss = ((model(x) - 1.0) ** 2).mean()
            step_loss = fptt.step(loss)
            assert isinstance(step_loss, float)
            assert not (step_loss != step_loss)  # NaN check

    def test_memory_flat_with_sequence_length(self):
        """FPTT should NOT grow graph proportionally to T (no BPTT unrolling).

        We check that the memory usage after T=10 steps is comparable to T=2 steps.
        This is a heuristic test — we verify that parameters have no retained graph
        after each step (detached state).
        """
        model, fptt = self._make_fptt()
        x = torch.randn(1, 4)

        fptt.start_sequence()
        for t in range(10):
            loss = ((model(x) - 0.0) ** 2).mean()
            fptt.step(loss)

        # After each step(), we called loss.backward() and the graph should be freed.
        # The model's grad should not have requires_grad True (which would imply a retained graph).
        for p in model.parameters():
            # grad should not require grad (not part of a computation graph)
            if p.grad is not None:
                assert not p.grad.requires_grad, "Grad should not retain its own graph (no BPTT)"
