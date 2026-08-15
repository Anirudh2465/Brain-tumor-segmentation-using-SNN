"""
metrics.py — Dice and NLL evaluation metrics for brain tumor segmentation.

Metrics match Tables 1 & 2 of arXiv:2601.16652v1.

Dice score: computed per class (ET, TC, WT) on binary predictions vs. targets.
NLL: standard binary NLL per voxel per class, averaged over subjects.
"""

from __future__ import annotations

import numpy as np
import torch


# ── Dice ──────────────────────────────────────────────────────────────────────

def dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Compute Dice score per class.

    Args:
        pred: Binary predictions, shape (B_or_T, 3, H, W) or (3, H, W).
              Should be 0.0/1.0 (thresholded).
        target: Binary ground truth, same shape as pred.
        eps: Smoothing constant.

    Returns:
        Dice score tensor of shape (3,) — one value per class [ET, TC, WT].
    """
    # Flatten spatial dims: (*, C, N) where N = prod of spatial dims
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    # Sum over batch and spatial dims
    intersection = (pred * target).sum(dim=(0, 2, 3))  # (C,)
    denom = pred.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3))  # (C,)
    return (2.0 * intersection + eps) / (denom + eps)  # (C,)


def dice_score_numpy(
    pred: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Dice score computation on numpy arrays.

    Args:
        pred: Binary predictions, shape (3, H, W, D) or (3, H, W).
        target: Binary ground truth, same shape.
        eps: Smoothing constant.

    Returns:
        Dice scores array of shape (3,) — [ET, TC, WT].
    """
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    # Sum over all spatial dims (everything except class axis 0)
    spatial_axes = tuple(range(1, pred.ndim))
    intersection = (pred * target).sum(axis=spatial_axes)  # (3,)
    denom = pred.sum(axis=spatial_axes) + target.sum(axis=spatial_axes)  # (3,)
    return (2.0 * intersection + eps) / (denom + eps)  # (3,)


# ── NLL ───────────────────────────────────────────────────────────────────────

def nll_score(
    pred_probs: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-7,
) -> np.ndarray:
    """Binary NLL per class, averaged over spatial voxels.

    NLL = -mean( y·log(p) + (1-y)·log(1-p) )

    Args:
        pred_probs: Probability predictions in [0, 1], shape (3, H, W, D) or (3, ...).
        target: Binary ground truth, same shape as pred_probs.
        eps: Clipping constant to avoid log(0).

    Returns:
        NLL array of shape (3,) — one value per class [ET, TC, WT].
    """
    pred_probs = np.clip(pred_probs.astype(np.float64), eps, 1.0 - eps)
    target = target.astype(np.float64)
    spatial_axes = tuple(range(1, pred_probs.ndim))
    nll = -(target * np.log(pred_probs) + (1.0 - target) * np.log(1.0 - pred_probs))
    return nll.mean(axis=spatial_axes)  # (3,)


# ── Aggregation helpers ────────────────────────────────────────────────────────

def aggregate_fold_metrics(
    per_subject_metrics: list[dict],
) -> dict[str, float]:
    """Compute mean and std of per-subject metrics across a fold.

    Args:
        per_subject_metrics: List of dicts, one per subject.
                             Each dict should have consistent keys.

    Returns:
        Dict with '{key}_mean' and '{key}_std' entries.
    """
    if not per_subject_metrics:
        return {}
    keys = per_subject_metrics[0].keys()
    result = {}
    for k in keys:
        vals = np.array([m[k] for m in per_subject_metrics], dtype=np.float64)
        result[f"{k}_mean"] = float(vals.mean())
        result[f"{k}_std"] = float(vals.std())
    return result
