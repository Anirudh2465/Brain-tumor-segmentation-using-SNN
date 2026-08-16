"""
flops.py — Spike-rate-aware FLOPs counter (Table 3 reproduction).

Standard FLOP counters (fvcore, ptflops) count *dense* operations.
For SNNs, effective computation scales with the spike rate (fraction of neurons
that fire at each time step). This module measures:

    effective_FLOPs = dense_FLOPs × mean_spike_rate

following the "synaptic operations" convention from Yin/Corradi/Bohté (NMI 2023).
See deviation D6 in DEVIATIONS.md.

Usage:
    from spiking_useg.efficiency.flops import measure_snn_flops
    result = measure_snn_flops(model, input_shape=(4, 160, 192))
    print(result)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional

try:
    from fvcore.nn import FlopCountAnalysis
    HAS_FVCORE = True
except ImportError:
    HAS_FVCORE = False


def _get_dense_flops(
    model: nn.Module,
    dummy_input: torch.Tensor,
) -> int:
    """Compute dense FLOPs for a single forward pass using fvcore.

    Args:
        model: Any nn.Module.
        dummy_input: A single input tensor (batch of 1).

    Returns:
        Total dense FLOPs as integer.
    """
    if not HAS_FVCORE:
        raise ImportError("fvcore is required for FLOPs counting. `pip install fvcore`")
    flop_counter = FlopCountAnalysis(model, dummy_input)
    flop_counter.unsupported_ops_warnings(False)
    flop_counter.uncalled_modules_warnings(False)
    return flop_counter.total()


def measure_snn_flops(
    model: nn.Module,
    input_shape: tuple[int, int, int],
    num_timesteps: int = 10,
    device: Optional[torch.device] = None,
    dataloader = None,
) -> dict:
    """Measure spike-rate-aware FLOPs for the SpikingUSegNet."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device).eval()

    # ── Collect spike rates via forward hooks ─────────────────────────────
    from spiking_useg.models.neurons import PLIFLayer
    spike_rates: list[float] = []
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            # out: spike tensor (B, C, H, W)
            rate = out.float().mean().item()
            spike_rates.append(rate)
        return hook

    for name, module in model.named_modules():
        if isinstance(module, PLIFLayer):
            h = module.register_forward_hook(make_hook(name))
            hooks.append(h)

    model.reset_states()
    C, H, W = input_shape
    with torch.no_grad():
        if dataloader is not None:
            batch = next(iter(dataloader))
            images = batch["images"].to(device) # (B, T, C, H, W)
            B, T_batch, C_b, H_b, W_b = images.shape
            for t in range(T_batch):
                model(images[:, t])
            num_timesteps = T_batch
        else:
            for _ in range(num_timesteps):
                dummy = torch.randn(1, C, H, W, device=device)
                model(dummy)

    for h in hooks:
        h.remove()

    mean_spike_rate = float(sum(spike_rates) / len(spike_rates)) if spike_rates else 1.0

    # ── Dense FLOPs via fvcore ─────────────────────────────────────────────
    model.reset_states()
    dummy_single = torch.randn(1, C, H, W, device=device)
    try:
        dense_flops = _get_dense_flops(model, dummy_single)
    except ImportError:
        dense_flops = -1
        print("Warning: fvcore not installed — dense FLOPs set to -1.")

    # ── Effective FLOPs ────────────────────────────────────────────────────
    effective_per_step = dense_flops * mean_spike_rate if dense_flops > 0 else -1
    effective_per_volume = effective_per_step * num_timesteps if effective_per_step > 0 else -1

    return {
        "dense_flops_per_step": dense_flops,
        "mean_spike_rate": mean_spike_rate,
        "effective_flops_per_step": effective_per_step,
        "effective_flops_per_volume": effective_per_volume,
    }
