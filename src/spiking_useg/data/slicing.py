"""
slicing.py — Convert preprocessed 3D volumes into ordered 2D slice sequences.

Each view produces a [T, 4, H, W] tensor where:
  - T = number of slices (= number of SNN time steps)
  - 4 = MRI modality channels (T1n, T1c, T2w, T2f)
  - H, W = spatial slice dimensions

View definitions (crop shape 160×192×152 = H×W×D):
  - Sagittal  (axis 0): T=160, each slice shape (4, 192, 152)
  - Coronal   (axis 1): T=192, each slice shape (4, 160, 152)
  - Axial     (axis 2): T=152, each slice shape (4, 160, 192)  [ASSUMPTION D2]
"""

from __future__ import annotations

from typing import Literal

import numpy as np

# View → (slice axis, num_slices, (H, W))
VIEW_CONFIG: dict[str, tuple[int, int, tuple[int, int]]] = {
    "sagittal": (0, 160, (192, 152)),
    "coronal":  (1, 192, (160, 152)),
    "axial":    (2, 152, (160, 192)),
}

ViewName = Literal["sagittal", "coronal", "axial"]


def slice_volume(
    data: np.ndarray,
    view: ViewName,
) -> np.ndarray:
    """Convert a 4-channel 3D volume into an ordered sequence of 2D slices.

    Args:
        data: float32 ndarray of shape (4, 160, 192, 152) — (C, H, W, D).
        view: One of 'sagittal', 'coronal', 'axial'.

    Returns:
        float32 ndarray of shape (T, 4, slice_H, slice_W).
    """
    if view not in VIEW_CONFIG:
        raise ValueError(f"Unknown view '{view}'. Choose from {list(VIEW_CONFIG.keys())}.")

    axis, T, (sH, sW) = VIEW_CONFIG[view]

    # data shape: (C, H, W, D) = (C, axis0, axis1, axis2)
    # We want to iterate along `axis` (in the spatial dims, which are axes 1,2,3 in data)
    spatial_axis = axis + 1  # +1 because data has C as axis 0

    # Move the slice axis to position 1: (C, T, sH, sW) → (T, C, sH, sW)
    # np.moveaxis(data, spatial_axis, 1) gives (C, T, ...) → we need (T, C, ...)
    moved = np.moveaxis(data, spatial_axis, 0)  # (T, C, ...)
    # moved shape: (T, C, dim_a, dim_b) in some order; need to verify sH×sW
    # Remaining spatial dims are those not equal to `axis`
    # After moveaxis, moved has shape (T, C, d_a, d_b)
    assert moved.shape[0] == T, f"Expected T={T}, got {moved.shape[0]} for view '{view}'"
    assert moved.shape[1] == 4, f"Expected 4 channels, got {moved.shape[1]}"
    assert moved.shape[2] == sH and moved.shape[3] == sW, (
        f"Expected slice shape ({sH},{sW}), got ({moved.shape[2]},{moved.shape[3]}) for view '{view}'"
    )
    return moved.astype(np.float32)


def slice_segmentation(
    seg: np.ndarray,
    view: ViewName,
) -> np.ndarray:
    """Convert a 3D segmentation volume into an ordered sequence of 2D label maps.

    Args:
        seg: int32 ndarray of shape (160, 192, 152) — (H, W, D).
        view: One of 'sagittal', 'coronal', 'axial'.

    Returns:
        int32 ndarray of shape (T, sH, sW) — raw BraTS label values {0,1,2,3}.
    """
    if view not in VIEW_CONFIG:
        raise ValueError(f"Unknown view '{view}'. Choose from {list(VIEW_CONFIG.keys())}.")

    axis, T, (sH, sW) = VIEW_CONFIG[view]

    # Move slice axis to front: (H, W, D) → (T, sH, sW)
    moved = np.moveaxis(seg, axis, 0)
    assert moved.shape == (T, sH, sW), (
        f"Expected ({T},{sH},{sW}), got {moved.shape} for view '{view}'"
    )
    return moved.astype(np.int32)


def derive_binary_targets(seg_slice: np.ndarray) -> np.ndarray:
    """Convert a raw BraTS label map to 3 binary target channels (ET, TC, WT).

    BraTS label values:
        0 = Background
        1 = NCR (Necrotic Core)
        2 = ED  (Peritumoral Edema)
        3 = ET  (Enhancing Tumor)

    Hierarchical regions:
        ET = label 3
        TC = labels 1 + 3 (NCR + ET)
        WT = labels 1 + 2 + 3 (all non-background)

    Args:
        seg_slice: int32 ndarray of shape (sH, sW).

    Returns:
        float32 ndarray of shape (3, sH, sW) — [ET, TC, WT] binary maps.
    """
    et = (seg_slice == 3).astype(np.float32)
    tc = ((seg_slice == 1) | (seg_slice == 3)).astype(np.float32)
    wt = (seg_slice > 0).astype(np.float32)
    return np.stack([et, tc, wt], axis=0)  # (3, sH, sW)
