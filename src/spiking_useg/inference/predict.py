"""
predict.py — Single-view full-volume inference.

Runs a trained SpikingUSegNet on a complete subject's slice sequence for
one anatomical view, and reassembles the per-slice predictions into a 3D
probability volume.

Output volume shape: (3, H, W, D) in the canonical (C, H, W, D) frame,
where C=3 corresponds to [ET, TC, WT].

The "canonical frame" is the preprocessed volume dimensions: (H=160, W=192, D=152).
Each view returns a 3D volume permuted back to this canonical orientation:
  - Axial   (axis=2): slices along D → output (3, H, W, D) directly
  - Coronal (axis=1): slices along W → need to permute back
  - Sagittal(axis=0): slices along H → need to permute back
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from spiking_useg.data.slicing import ViewName, VIEW_CONFIG, slice_volume


@torch.no_grad()
def predict_volume(
    model: nn.Module,
    data: np.ndarray,
    view: ViewName,
    device: torch.device,
    batch_size: int = 1,
) -> np.ndarray:
    """Run inference on one subject's preprocessed volume for a given view.

    Args:
        model: Trained SpikingUSegNet (already on `device`).
        data: Preprocessed volume, float32, shape (4, 160, 192, 152) = (C, H, W, D).
        view: Anatomical view to use.
        device: Inference device.
        batch_size: Number of slices to process at once (use 1 for full-sequence SNN mode).

    Returns:
        Probability volume, float32, shape (3, 160, 192, 152) = (ET/TC/WT, H, W, D)
        in the canonical orientation.
    """
    model.eval()

    # Get slice sequence: (T, 4, sH, sW)
    slices = slice_volume(data, view)   # (T, 4, sH, sW)
    T = slices.shape[0]

    if hasattr(model, 'module'):
        model.module.reset_states()
    else:
        model.reset_states()

    axis, _, (sH, sW) = VIEW_CONFIG[view]

    # Run slice-by-slice (preserving SNN temporal order)
    pred_slices: list[np.ndarray] = []
    for t in range(T):
        x_t = torch.from_numpy(slices[t]).unsqueeze(0).to(device)  # (1, 4, sH, sW)
        pred_t = model(x_t)   # (1, 3, sH, sW) logits
        pred_t = pred_t.sigmoid() # Convert to probabilities
        pred_slices.append(pred_t.squeeze(0).cpu().numpy())  # (3, sH, sW)

    # Stack: (T, 3, sH, sW)
    pred_seq = np.stack(pred_slices, axis=0)  # (T, 3, sH, sW)

    # Permute back to canonical (3, H=160, W=192, D=152) frame
    # For each view, we moved axis `axis` to position 0 (T-axis); undo that.
    # pred_seq axes: (T, C, dim_a, dim_b)
    # We want (C, 160, 192, 152)
    # First put C first: (C, T, dim_a, dim_b)
    pred_c_first = np.moveaxis(pred_seq, 1, 0)  # (3, T, sH, sW)

    # Now move T back to its original spatial axis position
    # spatial_axis = axis+1 (because C is now axis 0)
    # Currently T is axis 1; want it at axis `axis+1`
    spatial_axis = axis + 1
    pred_canonical = np.moveaxis(pred_c_first, 1, spatial_axis)  # (3, H, W, D)

    return pred_canonical.astype(np.float32)
