"""
neurons.py — PLIF (Parametric Leaky Integrate-and-Fire) neuron layer.

Implements the spiking neuron model from Section 3.1 of arXiv:2601.16652v1.

Membrane dynamics (discrete time):
    u_t = λ · u_{t-1} + I_t − θ · s_{t-1}
    s_t = H(u_t − θ)

where:
    λ = sigmoid(τ_param)  — learnable decay per layer (one scalar per layer)
    I_t                   — input current (pre-activation from conv/linear layer)
    θ                     — fixed firing threshold (default 1.0, deviation D3)
    s_{t-1}               — spike from previous time step (soft-reset term)
    H(·)                  — Heaviside with arctan surrogate gradient

Design choices (deviation D3, D4):
    - τ_param is a *single learnable scalar* per PLIFLayer, not per neuron.
      This matches the paper ("one τ per layer").
    - θ is fixed at 1.0 (not learnable) unless overridden.
    - State (u, s) is stored as instance buffers and MUST be reset between
      subject sequences by calling `reset_state()`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spiking_useg.models.surrogate import heaviside_arctan


class PLIFLayer(nn.Module):
    """Parametric Leaky Integrate-and-Fire neuron layer.

    This is NOT a single neuron — it processes a spatial feature map of
    shape (B, C, H, W) and applies the LIF dynamics element-wise.
    The learnable parameter τ_param is a single scalar for all neurons in this layer.

    Args:
        num_channels: Number of feature channels (C). Kept for documentation
            purposes; dynamics are applied channel/element-wise regardless.
        threshold: Fixed firing threshold θ (default 1.0).
        tau_init: Initial value of τ_param (λ = sigmoid(τ_param); default 0.0 → λ=0.5).
    """

    def __init__(
        self,
        num_channels: int,
        threshold: float = 1.0,
        tau_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.threshold = threshold
        # Single learnable decay parameter per layer
        self.tau_param = nn.Parameter(torch.tensor(tau_init, dtype=torch.float32))
        # Hidden state buffers — initialized lazily on first forward call
        self._u: torch.Tensor | None = None   # membrane potential
        self._s: torch.Tensor | None = None   # previous spike

    @property
    def decay(self) -> torch.Tensor:
        """Membrane decay λ = sigmoid(τ_param), guaranteed in (0, 1)."""
        return torch.sigmoid(self.tau_param)

    def reset_state(self) -> None:
        """Reset membrane potential and spike history to zero.

        Must be called at the start of each new subject sequence.
        """
        self._u = None
        self._s = None

    def forward(self, I_t: torch.Tensor) -> torch.Tensor:
        """One time-step forward pass.

        Args:
            I_t: Input current tensor of shape (B, C, H, W) or (B, C) for 1D.

        Returns:
            Spike tensor s_t, same shape as I_t, values in {0.0, 1.0}.
        """
        lam = self.decay  # scalar

        if self._u is None:
            # First time step — initialize state to zeros
            self._u = torch.zeros_like(I_t)
            self._s = torch.zeros_like(I_t)

        # Membrane update: u_t = λ·u_{t-1} + I_t − θ·s_{t-1}
        u_t = lam * self._u + I_t - self.threshold * self._s

        # Spike generation with surrogate gradient
        s_t = heaviside_arctan(u_t - self.threshold)

        # Store state (detach from graph — FPTT handles inter-step gradients)
        self._u = u_t.detach()
        self._s = s_t.detach()

        return s_t

    def extra_repr(self) -> str:
        return f"threshold={self.threshold}, tau_init={self.tau_param.data.item():.3f}"
