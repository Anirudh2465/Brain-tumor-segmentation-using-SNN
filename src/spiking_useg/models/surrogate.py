"""
surrogate.py — Arctan surrogate gradient for spiking Heaviside function.

The forward pass returns the true Heaviside H(x) (discontinuous step).
The backward pass returns the gradient of the smooth arctan approximation:

    σ(x) = (1/π) arctan(πx) + 1/2
    σ'(x) = 1 / (1 + (πx)²)

This is the surrogate gradient used in the paper (Section 3.1).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.autograd import Function


class _ArctanSurrogate(Function):
    """Custom autograd Function: Heaviside forward + arctan surrogate backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        """Forward: standard Heaviside (spike if x > 0)."""
        ctx.save_for_backward(x)
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """Backward: σ'(x) = 1 / (1 + (πx)²)."""
        (x,) = ctx.saved_tensors
        surrogate_grad = 1.0 / (1.0 + (torch.pi * x) ** 2)
        return grad_output * surrogate_grad


def heaviside_arctan(x: torch.Tensor) -> torch.Tensor:
    """Apply Heaviside with arctan surrogate gradient.

    Args:
        x: Membrane potential offset (u - threshold).

    Returns:
        Binary spike tensor (0.0 or 1.0), float.
    """
    return _ArctanSurrogate.apply(x)
