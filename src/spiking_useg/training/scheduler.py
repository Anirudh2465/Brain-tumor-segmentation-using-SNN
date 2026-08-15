"""
scheduler.py — LR scheduler wrapper for FPTT training.

Uses ReduceLROnPlateau on validation Dice (paper Section 4.3).
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau


def build_scheduler(optimizer: Optimizer, patience: int = 5, factor: float = 0.5) -> ReduceLROnPlateau:
    """Build a ReduceLROnPlateau scheduler.

    Reduces LR when validation Dice stops improving.

    Args:
        optimizer: The base optimizer (Adam).
        patience: Epochs without improvement before LR reduction (default 5).
        factor: LR reduction factor (default 0.5).

    Returns:
        ReduceLROnPlateau scheduler instance.
    """
    return ReduceLROnPlateau(
        optimizer,
        mode="max",       # maximize Dice score
        patience=patience,
        factor=factor,
        min_lr=1e-6,
    )
