"""
predict_slice.py — Inference utilities for single-slice axial mode.

Functions:
    predict_single_slice  — run the model on one (4, H, W) axial slice
    find_best_slice       — find the axial slice index with the most tumour area
    predict_best_slice    — convenience: find the best slice + run inference on it
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


# ── Slice selection ────────────────────────────────────────────────────────────

def find_best_slice(seg: np.ndarray) -> int:
    """Find the axial slice index with the largest tumour area.

    Args:
        seg: Integer segmentation volume, shape (H, W, D) with BraTS labels
             {0, 1, 2, 3}.  D = 152 for the axial axis after central crop.

    Returns:
        Axial slice index (along the D axis, i.e. axis 2) that contains the
        most non-background tumour voxels.  Falls back to D // 2 if no tumour
        is found at all.
    """
    # Sum non-background labels along each axial slice: result shape (D,)
    tumor_per_slice = (seg > 0).sum(axis=(0, 1))   # (D,)
    if tumor_per_slice.max() == 0:
        return seg.shape[2] // 2                   # no tumour found → middle
    return int(np.argmax(tumor_per_slice))


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_single_slice(
    model: nn.Module,
    image_slice: np.ndarray,
    device: torch.device,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference on one axial MRI slice.

    Args:
        model: Trained SpikingUSegNet, already moved to `device` and in eval mode.
        image_slice: float32 ndarray of shape (4, H, W) — one axial slice,
                     4 MRI modalities, values in [0, 1].
        device: Inference device.
        threshold: Binarisation threshold (default 0.5).

    Returns:
        Tuple (prob_map, seg_mask):
            prob_map  : float32 ndarray (3, H, W) — raw probabilities [0, 1]
                        channels = [ET, TC, WT]
            seg_mask  : bool ndarray (3, H, W) — thresholded binary masks
    """
    model.eval()

    # Reset PLIF state — each slice is an independent T=1 SNN step
    if hasattr(model, "module"):
        model.module.reset_states()
    else:
        model.reset_states()

    # (4, H, W) → (1, 4, H, W)
    x = torch.from_numpy(image_slice.astype(np.float32)).unsqueeze(0).to(device)

    prob = model(x)                         # (1, 3, H, W)
    prob_np = prob.squeeze(0).cpu().numpy() # (3, H, W)
    seg_np  = prob_np > threshold           # (3, H, W) — bool

    return prob_np, seg_np


def predict_best_slice(
    model: nn.Module,
    data: np.ndarray,
    seg: Optional[np.ndarray],
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    """Find the most informative axial slice, run inference, return results.

    Args:
        model: Trained SpikingUSegNet in eval mode.
        data: Preprocessed volume, float32, shape (4, H, W, D) = (4, 160, 192, 152).
        seg:  Ground-truth label map, int32, shape (H, W, D) = (160, 192, 152).
              Pass None for unlabelled data.
        device: Inference device.
        threshold: Binarisation threshold.

    Returns:
        Dict with keys:
            'slice_idx'   : int  — chosen axial slice index
            'image_slice' : float32 (4, H, W) — raw MRI slice
            'gt_seg_slice': int32   (H, W)     — GT label map slice (or None)
            'prob_map'    : float32 (3, H, W)  — ET/TC/WT probabilities
            'pred_seg'    : bool    (3, H, W)  — ET/TC/WT binary predictions
            'pred_brats'  : int32   (H, W)     — BraTS label map reconstructed
                                                  from pred_seg
    """
    # ── Pick the best slice ───────────────────────────────────────────────
    if seg is not None:
        slice_idx = find_best_slice(seg)
    else:
        slice_idx = data.shape[3] // 2   # middle axial slice

    # ── Extract the 2D slice ──────────────────────────────────────────────
    image_slice = data[:, :, :, slice_idx]   # (4, H, W)

    gt_seg_slice: Optional[np.ndarray] = None
    if seg is not None:
        gt_seg_slice = seg[:, :, slice_idx]   # (H, W) with BraTS labels

    # ── Inference ─────────────────────────────────────────────────────────
    prob_map, pred_seg = predict_single_slice(
        model, image_slice, device, threshold=threshold
    )

    # ── Reconstruct BraTS label map from hierarchical predictions ─────────
    # Hierarchy: WT ⊃ TC ⊃ ET
    #   WT voxels → label 2 (ED baseline)
    #   TC voxels → label 1 (NCR, overwrites ED)
    #   ET voxels → label 3 (ET,  overwrites NCR)
    pred_brats = np.zeros(image_slice.shape[1:], dtype=np.int32)  # (H, W)
    pred_brats[pred_seg[2]] = 2  # WT → ED
    pred_brats[pred_seg[1]] = 1  # TC → NCR
    pred_brats[pred_seg[0]] = 3  # ET

    return {
        "slice_idx":    slice_idx,
        "image_slice":  image_slice,
        "gt_seg_slice": gt_seg_slice,
        "prob_map":     prob_map,
        "pred_seg":     pred_seg,
        "pred_brats":   pred_brats,
    }
