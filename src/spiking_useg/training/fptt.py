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
        self._params = list(self._param_groups())
        self._w_prev = [p.data.clone() for p in self._params]
        self._w_bar = [p.data.clone() for p in self._params]
        self._grad_accum = [torch.zeros_like(p.data) for p in self._params]
        self._initialized = True

    def _update_buffers(self) -> None:
        """Update FPTT state buffers after the optimizer step."""
        with torch.no_grad():
            w_t = [p.data for p in self._params]
            
            # ∇l_t = ∇l_{t-1} − α(w_t − w_{t-1})
            w_diff = torch._foreach_sub(w_t, self._w_prev)
            torch._foreach_add_(self._grad_accum, w_diff, alpha=-self.alpha)
            
            # w̄_{t+1} = ½(w_t + w̄_t)
            torch._foreach_add_(self._w_bar, w_t)
            torch._foreach_mul_(self._w_bar, 0.5)
            
            # w_prev = w_t
            self._w_prev = [w.clone() for w in w_t]

    def step(self, task_loss: torch.Tensor) -> float:
        """Perform one FPTT step."""
        if not self._initialized:
            self._init_buffers()

        self.base_optimizer.zero_grad()
        task_loss.backward()

        total_loss_val = task_loss.item()

        if self._t > 0:
            with torch.no_grad():
                # Manually compute and add regularizer gradients to avoid autograd overhead
                # target_w = w̄_t - (1/2α)∇l_{t-1}
                factor = 1.0 / (2.0 * self.alpha)
                grad_terms = torch._foreach_mul(self._grad_accum, factor)
                target_w = torch._foreach_sub(self._w_bar, grad_terms)
                
                # diff = w_t - target_w
                w_t = [p.data for p in self._params]
                diff = torch._foreach_sub(w_t, target_w)
                
                # reg_grads = α * diff
                reg_grads = torch._foreach_mul(diff, self.alpha)
                
                # Track regularization loss for logging: L_reg = (α/2) ||diff||^2
                reg_loss_val = (self.alpha / 2.0) * sum([d.pow(2).sum().item() for d in diff])
                total_loss_val += reg_loss_val
                
                # Add reg_grads to param.grad
                for i, p in enumerate(self._params):
                    if p.grad is not None:
                        p.grad.add_(reg_grads[i])

        # Gradient clipping
        params_with_grad = [p for p in self._params if p.grad is not None]
        if params_with_grad:
            nn.utils.clip_grad_norm_(params_with_grad, max_norm=self.grad_clip_norm)

        self.base_optimizer.step()

        self._update_buffers()
        self._t += 1

        return total_loss_val
