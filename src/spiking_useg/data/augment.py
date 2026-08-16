"""
augment.py — Slice-consistent augmentations for BraTS data.

IMPORTANT: Augmentations must be applied identically to ALL T slices in a
sequence so that the temporal ordering of the SNN input is preserved. Doing
random per-slice augmentation would corrupt the spatial coherence that the
recurrent spiking model relies on.

All functions operate on numpy arrays:
    images : float32 (T, 4, H, W)
    targets: float32 (T, 3, H, W)
"""

from __future__ import annotations

import numpy as np


class RandomHorizontalFlip:
    """Randomly flip all slices in the sequence horizontally (along W)."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(
        self,
        images: np.ndarray,
        targets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if np.random.rand() < self.p:
            images = images[:, :, :, ::-1].copy()
            targets = targets[:, :, :, ::-1].copy()
        return images, targets


class RandomVerticalFlip:
    """Randomly flip all slices in the sequence vertically (along H)."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(
        self,
        images: np.ndarray,
        targets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if np.random.rand() < self.p:
            images = images[:, :, ::-1, :].copy()
            targets = targets[:, :, ::-1, :].copy()
        return images, targets


class RandomIntensityJitter:
    """Add a small per-sequence Gaussian noise to image intensities.

    The same noise scale is used for all slices to avoid temporal
    inconsistencies — but individual voxel noise is i.i.d. per slice.
    """

    def __init__(self, std: float = 0.02) -> None:
        self.std = std

    def __call__(
        self,
        images: np.ndarray,
        targets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        noise = np.random.normal(0.0, self.std, size=images.shape).astype(np.float32)
        images = np.clip(images + noise, 0.0, 1.0)
        return images, targets


class Compose:
    """Compose multiple augmentation transforms."""

    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(
        self,
        images: np.ndarray,
        targets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        for t in self.transforms:
            images, targets = t(images, targets)
        return images, targets


def default_train_augmentation() -> Compose:
    """Standard augmentation pipeline for training."""
    return Compose([
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomIntensityJitter(std=0.01),
    ])
