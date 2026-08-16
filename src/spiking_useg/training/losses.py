"""
losses.py — Hybrid BCE + Dice loss for 3-head brain tumor segmentation.

From Section 3.4 of arXiv:2601.16652v1:
    L_total = 0.5 × L_BCE + 0.5 × L_Dice

Applied per-class (ET, TC, WT) and averaged (deviation D5: equal weighting).

Both pred and target should have values in [0, 1] and shape (B, 3, H, W) or
(B, 3) for aggregated cases.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Soft Dice loss for binary segmentation.

    Args:
        pred: Predicted probabilities, shape (...).
        target: Binary targets, same shape as pred.
        eps: Smoothing constant to avoid division by zero.

    Returns:
        Scalar Dice loss = 1 − Dice coefficient.
    """
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return 1.0 - (2.0 * intersection + eps) / (union + eps)


def hybrid_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Hybrid BCE + Dice loss, averaged over the 3 output classes."""
    num_classes = logits.shape[1]
    total_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    
    probs = torch.sigmoid(logits)

    for c in range(num_classes):
        logits_c = logits[:, c, ...]
        probs_c = probs[:, c, ...]
        tgt_c = target[:, c, ...]

        bce = F.binary_cross_entropy_with_logits(logits_c, tgt_c, reduction="mean")
        dice = dice_loss(probs_c, tgt_c, eps=eps)

        total_loss = total_loss + bce_weight * bce + dice_weight * dice

    return total_loss / num_classes
