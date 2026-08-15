"""
viz.py — Visualization utilities: tri-view MRI display and segmentation overlay.

Produces Figure 1-style tri-plane views (axial / coronal / sagittal) with
optional segmentation overlay.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


MODALITY_NAMES = ["T1n", "T1c", "T2w", "T2f (FLAIR)"]
LABEL_COLORS = {
    1: (1.0, 0.0, 0.0, 0.6),   # NCR  — red
    2: (1.0, 1.0, 0.0, 0.6),   # ED   — yellow
    3: (0.0, 1.0, 0.0, 0.6),   # ET   — green
}


def _get_mid_slices(volume: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the axial, coronal, and sagittal mid-slices of a (H, W, D) volume."""
    H, W, D = volume.shape
    axial = volume[:, :, D // 2]      # (H, W)
    coronal = volume[:, W // 2, :]    # (H, D)
    sagittal = volume[H // 2, :, :]  # (W, D)
    return axial, coronal, sagittal


def plot_triview(
    data: np.ndarray,
    seg: Optional[np.ndarray] = None,
    modality_idx: int = 1,
    title: str = "BraTS Subject",
    save_path: Optional[Path | str] = None,
) -> plt.Figure:
    """Plot axial / coronal / sagittal slices for one MRI modality.

    Args:
        data: Preprocessed volume, shape (4, H, W, D) — (C, H, W, D).
        seg: Optional segmentation mask, shape (H, W, D), integer labels.
        modality_idx: Channel index to display (0=T1n, 1=T1c, 2=T2w, 3=FLAIR).
        title: Figure title.
        save_path: If provided, save the figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    vol = data[modality_idx]  # (H, W, D)
    views = _get_mid_slices(vol)
    view_names = ["Axial (mid)", "Coronal (mid)", "Sagittal (mid)"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{title} — {MODALITY_NAMES[modality_idx]}", fontsize=14)

    seg_views = None
    if seg is not None:
        seg_views = _get_mid_slices(seg)

    for ax, view, name, i in zip(axes, views, view_names, range(3)):
        ax.imshow(view.T, cmap="gray", origin="lower")
        ax.set_title(name)
        ax.axis("off")

        if seg_views is not None:
            seg_v = seg_views[i]
            # Overlay each label in a different color
            h_img, w_img = view.T.shape
            overlay = np.zeros((h_img, w_img, 4))
            for label, color in LABEL_COLORS.items():
                mask = (seg_v.T == label)
                for c in range(4):
                    overlay[mask, c] = color[c]
            ax.imshow(overlay, origin="lower")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    return fig


def plot_segmentation_comparison(
    data: np.ndarray,
    gt_seg: np.ndarray,
    pred_seg: np.ndarray,
    title: str = "Prediction vs. Ground Truth",
    save_path: Optional[Path | str] = None,
) -> plt.Figure:
    """Side-by-side comparison of GT and predicted segmentation.

    Args:
        data: (4, H, W, D) MRI volume.
        gt_seg: (H, W, D) ground truth label map.
        pred_seg: (H, W, D) predicted label map.
        title: Figure title.
        save_path: Optional save path.

    Returns:
        Matplotlib Figure.
    """
    vol = data[1]  # Use T1c
    ax_vol, _, _ = _get_mid_slices(vol)
    ax_gt, _, _ = _get_mid_slices(gt_seg)
    ax_pred, _, _ = _get_mid_slices(pred_seg)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=14)

    axes[0].imshow(ax_vol.T, cmap="gray", origin="lower")
    axes[0].set_title("T1c (axial mid)")
    axes[0].axis("off")

    for ax, seg, name in [(axes[1], ax_gt, "Ground Truth"), (axes[2], ax_pred, "Prediction")]:
        ax.imshow(ax_vol.T, cmap="gray", origin="lower")
        overlay = np.zeros((*ax_vol.T.shape, 4))
        for label, color in LABEL_COLORS.items():
            mask = (seg.T == label)
            for c in range(4):
                overlay[mask, c] = color[c]
        ax.imshow(overlay, origin="lower")
        ax.set_title(name)
        ax.axis("off")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    return fig
