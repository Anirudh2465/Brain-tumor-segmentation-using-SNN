"""
layers.py — Spiking convolutional block: Conv2d → GroupNorm → Dropout → PLIF.

Each SpikingBlock processes one time-step's 2D feature map and contains its own
PLIFLayer whose state persists across time steps within a sequence.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spiking_useg.models.neurons import PLIFLayer


class SpikingBlock(nn.Module):
    """Single spiking convolutional block.

    Architecture per block (matches paper Section 3.1):
        Conv2d(same padding) → GroupNorm → Dropout → PLIF

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size (default 3×3).
        dropout: Dropout probability (default 0.1).
        groupnorm_groups: Number of groups for GroupNorm (deviation D4: 8).
        threshold: PLIF firing threshold θ (default 1.0).
        tau_init: Initial τ_param for PLIF (default 0.0 → λ=0.5).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
        groupnorm_groups: int = 8,
        threshold: float = 1.0,
        tau_init: float = 0.0,
    ) -> None:
        super().__init__()
        # Clamp groups to not exceed channels (handles small channel counts)
        groups = min(groupnorm_groups, out_channels)
        # GroupNorm requires channels to be divisible by groups
        while out_channels % groups != 0 and groups > 1:
            groups -= 1

        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,  # 'same' padding for odd kernel sizes
            bias=False,  # GroupNorm has affine params; no need for conv bias
        )
        self.norm = nn.GroupNorm(groups, out_channels)
        self.dropout = nn.Dropout2d(p=dropout)
        self.plif = PLIFLayer(
            num_channels=out_channels,
            threshold=threshold,
            tau_init=tau_init,
        )

    def reset_state(self) -> None:
        """Reset the PLIF neuron state — call at start of each subject sequence."""
        self.plif.reset_state()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process one time-step slice.

        Args:
            x: Input tensor (B, in_channels, H, W).

        Returns:
            Spike tensor (B, out_channels, H, W) — values in {0.0, 1.0} (approx).
        """
        x = self.conv(x)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.plif(x)
        return x
