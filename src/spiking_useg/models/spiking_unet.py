"""
spiking_unet.py — Full Spiking U-Seg-Net architecture.

Architecture summary (Section 3.1, arXiv:2601.16652v1):
  - U-Net encoder–decoder with skip connections
  - SINGLE SpikingBlock per resolution level (not 2× as in vanilla U-Net)
  - Channel progression: 4 → 32 → 64 → 128 → 128 (bottleneck)
                         → 128 → 128 → 64 → 32 → (3 outputs) [deviation D1]
  - Downsampling: MaxPool2d(2×2)
  - Upsampling: ConvTranspose2d(2×2, stride=2)
  - Skip connections: concat along channel dim at matching resolutions
  - Final readout: 1×1 Conv2d → 3 output channels (ET, TC, WT) — real-valued,
    followed by Sigmoid (no PLIF on the final layer — "integrator" in paper)

The model is designed to be called ONE SLICE AT A TIME:
    pred_t = model(x_t)   # x_t: (B, 4, H, W)
    pred_t: (B, 3, H, W)  # probabilities in [0,1]

Internal spiking states (membrane potentials) persist across calls.
Call model.reset_states() at the start of each new subject sequence.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from spiking_useg.models.layers import SpikingBlock


class SpikingUSegNet(nn.Module):
    """Spiking U-Seg-Net for 2D slice-by-slice brain tumor segmentation.

    Args:
        in_channels: Number of MRI modality channels (default 4).
        num_classes: Number of segmentation output classes (default 3: ET, TC, WT).
        encoder_channels: Feature channels at each encoder resolution.
        decoder_channels: Feature channels at each decoder resolution.
        dropout: Dropout probability in each SpikingBlock.
        groupnorm_groups: GroupNorm group count.
        threshold: PLIF firing threshold θ.
        tau_init: Initial τ_param for PLIF layers.
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 3,
        encoder_channels: tuple[int, ...] = (32, 64, 128, 128),
        decoder_channels: tuple[int, ...] = (128, 128, 64, 32),
        dropout: float = 0.1,
        groupnorm_groups: int = 8,
        threshold: float = 1.0,
        tau_init: float = 0.0,
    ) -> None:
        super().__init__()

        block_kwargs = dict(
            dropout=dropout,
            groupnorm_groups=groupnorm_groups,
            threshold=threshold,
            tau_init=tau_init,
        )

        # ── Input projection: 4 → 32 ─────────────────────────────────────
        self.input_proj = SpikingBlock(
            in_channels, encoder_channels[0], **block_kwargs
        )

        # ── Encoder (3 downsampling levels) ──────────────────────────────
        # enc1: 32 → 64 (after pool)
        # enc2: 64 → 128 (after pool)
        # enc3: 128 → 128 (bottleneck, after pool)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc1 = SpikingBlock(encoder_channels[0], encoder_channels[1], **block_kwargs)
        self.enc2 = SpikingBlock(encoder_channels[1], encoder_channels[2], **block_kwargs)
        self.enc3 = SpikingBlock(encoder_channels[2], encoder_channels[3], **block_kwargs)

        # ── Decoder (3 upsampling levels) ────────────────────────────────
        # up3: upsample + concat(skip enc2) + block
        # up2: upsample + concat(skip enc1) + block
        # up1: upsample + concat(skip input_proj) + block

        # After upsample and concat, input channels = decoder_ch + skip_ch
        self.up3_conv = nn.ConvTranspose2d(
            encoder_channels[3], decoder_channels[0], kernel_size=2, stride=2
        )
        # concat with enc2 output → in_channels = dec0 + enc2_ch = 128+128=256
        self.dec3 = SpikingBlock(
            decoder_channels[0] + encoder_channels[2], decoder_channels[1], **block_kwargs
        )

        self.up2_conv = nn.ConvTranspose2d(
            decoder_channels[1], decoder_channels[2], kernel_size=2, stride=2
        )
        # concat with enc1 output → in_channels = dec2 + enc1_ch = 64+64=128
        self.dec2 = SpikingBlock(
            decoder_channels[2] + encoder_channels[1], decoder_channels[2], **block_kwargs
        )

        self.up1_conv = nn.ConvTranspose2d(
            decoder_channels[2], decoder_channels[3], kernel_size=2, stride=2
        )
        # concat with input_proj output → in_channels = dec3 + enc0_ch = 32+32=64
        self.dec1 = SpikingBlock(
            decoder_channels[3] + encoder_channels[0], decoder_channels[3], **block_kwargs
        )

        # ── Output head (real-valued, no PLIF — "integrator") ────────────
        self.output_conv = nn.Conv2d(decoder_channels[3], num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    # ── State management ──────────────────────────────────────────────────

    def _all_spiking_blocks(self) -> list[SpikingBlock]:
        """Return all SpikingBlock modules in the network."""
        return [
            m for m in self.modules() if isinstance(m, SpikingBlock)
        ]

    def reset_states(self) -> None:
        """Reset all PLIF neuron membrane states to zero.

        MUST be called at the start of each new subject sequence (before
        iterating through its slices). Failure to do so will cause state
        contamination from the previous subject.
        """
        for block in self._all_spiking_blocks():
            block.reset_state()

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, x_t: torch.Tensor) -> torch.Tensor:
        """Process one 2D slice (one SNN time step).

        Args:
            x_t: Input tensor of shape (B, 4, H, W) — one slice from a subject.

        Returns:
            Prediction tensor of shape (B, 3, H, W) — probabilities in [0, 1]
            for [ET, TC, WT] at each voxel in this slice.
        """
        # ── Encoder ───────────────────────────────────────────────────────
        # Resolution: (H, W)
        e0 = self.input_proj(x_t)      # (B, 32, H, W)

        # Resolution: (H/2, W/2)
        e1 = self.enc1(self.pool(e0))  # (B, 64, H/2, W/2)

        # Resolution: (H/4, W/4)
        e2 = self.enc2(self.pool(e1))  # (B, 128, H/4, W/4)

        # Resolution: (H/8, W/8)  — bottleneck
        e3 = self.enc3(self.pool(e2))  # (B, 128, H/8, W/8)

        # ── Decoder ───────────────────────────────────────────────────────
        # Up to (H/4, W/4)
        d3 = self.up3_conv(e3)                        # (B, 128, H/4, W/4)
        d3 = self._pad_and_cat(d3, e2)               # (B, 256, H/4, W/4)
        d3 = self.dec3(d3)                            # (B, 128, H/4, W/4)

        # Up to (H/2, W/2)
        d2 = self.up2_conv(d3)                        # (B, 64, H/2, W/2)
        d2 = self._pad_and_cat(d2, e1)               # (B, 128, H/2, W/2)
        d2 = self.dec2(d2)                            # (B, 64, H/2, W/2)

        # Up to (H, W)
        d1 = self.up1_conv(d2)                        # (B, 32, H, W)
        d1 = self._pad_and_cat(d1, e0)               # (B, 64, H, W)
        d1 = self.dec1(d1)                            # (B, 32, H, W)

        # ── Output ────────────────────────────────────────────────────────
        out = self.output_conv(d1)    # (B, 3, H, W) — real-valued logits
        return out

    @staticmethod
    def _pad_and_cat(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Pad x to match skip's spatial dims, then concatenate along channel axis.

        This handles the occasional off-by-one from odd spatial dimensions
        after maxpool + transposed conv, which is common in U-Nets.

        Args:
            x: Upsampled tensor (B, C1, H', W').
            skip: Skip connection tensor (B, C2, H, W) where H >= H', W >= W'.

        Returns:
            Concatenated tensor (B, C1+C2, H, W).
        """
        diff_h = skip.shape[2] - x.shape[2]
        diff_w = skip.shape[3] - x.shape[3]
        # Pad on right/bottom to match
        if diff_h > 0 or diff_w > 0:
            x = torch.nn.functional.pad(x, [0, diff_w, 0, diff_h])
        return torch.cat([x, skip], dim=1)
