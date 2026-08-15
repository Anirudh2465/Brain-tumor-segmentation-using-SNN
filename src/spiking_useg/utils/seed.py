"""
seed.py — Reproducibility seeding for all randomness sources.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Set seeds for Python, NumPy, and PyTorch for reproducibility.

    Also sets CUBLAS_WORKSPACE_CONFIG for deterministic CUDA operations.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Enable deterministic algorithms where possible
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
