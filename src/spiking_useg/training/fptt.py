"""
fptt.py — FPTT (Forward Propagation Through Time) optimizer wrapper.

Implements the FPTT update rule from Section 3.2 of arXiv:2601.16652v1,
which in turn follows Kag & Saligrama (2021) and Yin et al. (NMI 2023).

Key property: No BPTT — gradients are only propagated through the CURRENT
time step, not unrolled across all T slices. This gives O(1) memory scaling
in sequence length T (versus O(T) for standard BPTT).

FPTT Update Rule (per time step t):
─────────────────────────────────────────────────────────────────────────────
Let:
    w_t   = current model weights
    w̄_t   = running average of past weights (exponential moving average)
    ∇l_t  = accumulated gradient signal

At each time step t:
  1. Compute task loss L(y_t, ŷ_t).
  2. Add regularizer: R(w_t) = (α/2) ‖ w_t − (w̄_t − (1/2α)∇l_{t-1}) ‖²
  3. Total loss: Ω_t = L_t + R(w_t)
  4. Backprop only through current step.
  5. Update gradient accumulator:
         ∇l_t = ∇l_{t-1} − α(w_t − w_{t-1})
  6. Update weight average:
         w̄_{t+1} = ½(w_t + w̄_t)          [exponential mean approximation]
  7. Apply base optimizer (Adam) step on Ω_t's gradients.
─────────────────────────────────────────────────────────────────────────────

Usage:
    base_optim = torch.optim.Adam(model.parameters(), lr=0.001)
    fptt = FPTTOptimizer(base_optim, alpha=0.1)

    # Inside the per-slice loop:
    fptt.start_sequence()           # reset at start of each subject sequence
    for t, (x_t, y_t) in enumerate(slices):
        pred_t = model(x_t)
        loss_t = hybrid_loss(pred_t, y_t)
        fptt.step(loss_t)           # handles backward + FPTT correction + Adam
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


class FPTTOptimizer:
    """FPTT optimizer wrapper around a base optimizer (Adam).

    Args:
        base_optimizer: A standard PyTorch optimizer (e.g., Adam).
        alpha: FPTT regularization strength (paper: 0.1 optimal).
        grad_clip_norm: Max gradient norm for clipping (paper: 0.3).
    """

    def __init__(
        self,
        base_optimizer: Optimizer,
        alpha: float = 0.1,
        grad_clip_norm: float = 0.3,
    ) -> None:
        self.base_optimizer = base_optimizer
        self.alpha = alpha
        self.grad_clip_norm = grad_clip_norm

        # Per-parameter state buffers (allocated on first step)
        self._w_prev: dict[str, torch.Tensor] = {}    # w_{t-1}
        self._w_bar: dict[str, torch.Tensor] = {}     # running weight average
        self._grad_accum: dict[str, torch.Tensor] = {}  # ∇l_{t-1}
        self._initialized = False
        self._t = 0  # time step counter within current sequence

    def _param_groups(self):
        """Iterate over all (name, param) pairs in the model."""
        for group in self.base_optimizer.param_groups:
            for param in group["params"]:
                if param.requires_grad:
                    yield param

    def start_sequence(self) -> None:
        """Reset FPTT state at the start of a new subject sequence.

        This must be called before iterating through each subject's slices.
        It initializes/resets w̄ and ∇l buffers.
        """
        self._t = 0
        self._w_prev.clear()
        self._w_bar.clear()
        self._grad_accum.clear()
        self._initialized = False

    def _init_buffers(self) -> None:
        """Initialize FPTT buffers from current model parameters."""
        for param in self._param_groups():
            pid = id(param)
            self._w_prev[pid] = param.data.clone()
            self._w_bar[pid] = param.data.clone()
            self._grad_accum[pid] = torch.zeros_like(param.data)
        self._initialized = True

    def _compute_regularizer_loss(self) -> torch.Tensor:
        """Compute FPTT regularizer R(w_t).

        R(w_t) = (α/2) Σ ‖ w_t − (w̄_t − (1/2α)∇l_{t-1}) ‖²

        Returns:
            Scalar regularizer tensor.
        """
        reg = torch.tensor(0.0, device=next(self._param_groups()).device)
        for param in self._param_groups():
            pid = id(param)
            target_w = self._w_bar[pid] - (1.0 / (2.0 * self.alpha)) * self._grad_accum[pid]
            diff = param - target_w.detach()
            reg = reg + (self.alpha / 2.0) * (diff * diff).sum()
        return reg

    def _update_buffers(self) -> None:
        """Update FPTT state buffers after the optimizer step.

        Updates:
            ∇l_t = ∇l_{t-1} − α(w_t − w_{t-1})
            w̄_{t+1} = ½(w_t + w̄_t)
            w_prev = w_t
        """
        for param in self._param_groups():
            pid = id(param)
            w_t = param.data.clone()
            w_prev = self._w_prev[pid]
            # Update gradient accumulator
            self._grad_accum[pid] = self._grad_accum[pid] - self.alpha * (w_t - w_prev)
            # Update running average
            self._w_bar[pid] = 0.5 * (w_t + self._w_bar[pid])
            # Save current weights for next step
            self._w_prev[pid] = w_t

    def step(self, task_loss: torch.Tensor) -> float:
        """Perform one FPTT step (one time step / one slice).

        Args:
            task_loss: Scalar loss from the current time step's prediction.

        Returns:
            Total loss value (task + regularizer) as a Python float.
        """
        if not self._initialized:
            self._init_buffers()

        # Add FPTT regularizer to task loss
        if self._t > 0:
            reg = self._compute_regularizer_loss()
            total_loss = task_loss + reg
        else:
            total_loss = task_loss  # no regularizer at t=0 (no previous state)

        # Backward through current time step ONLY
        self.base_optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping (paper: max norm 0.3)
        params_with_grad = [p for p in self._param_groups() if p.grad is not None]
        if params_with_grad:
            nn.utils.clip_grad_norm_(params_with_grad, max_norm=self.grad_clip_norm)

        # Base optimizer step (Adam)
        self.base_optimizer.step()

        # Update FPTT state buffers
        self._update_buffers()
        self._t += 1

        return total_loss.item()
