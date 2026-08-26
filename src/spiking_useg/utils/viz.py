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


# ── Single-slice overlay (single-slice axial mode) ────────────────────────────

from pathlib import Path as _Path
from typing import Optional as _Optional

# Colour map for the 3 tumour regions shown in the overlay panels.
# Keys are BraTS integer labels; values are RGBA tuples.
_OVERLAY_COLORS: dict[int, tuple] = {
    1: (1.0, 0.2, 0.2, 0.65),   # NCR / TC core — red
    2: (1.0, 1.0, 0.2, 0.55),   # ED  / WT rim  — yellow
    3: (0.2, 1.0, 0.2, 0.70),   # ET             — green
}


def _seg_to_rgba(seg_2d: np.ndarray, shape_hw: tuple) -> np.ndarray:
    """Convert an integer label map (H, W) to an RGBA overlay (H, W, 4)."""
    overlay = np.zeros((*shape_hw, 4), dtype=np.float32)
    for label, color in _OVERLAY_COLORS.items():
        mask = seg_2d == label
        for c in range(4):
            overlay[mask, c] = color[c]
    return overlay


def plot_slice_overlay(
    image_slice: np.ndarray,
    gt_seg_slice: _Optional[np.ndarray],
    pred_brats: np.ndarray,
    case_id: str = "",
    slice_idx: int = -1,
    save_path: _Optional[_Path | str] = None,
    modality_idx: int = 1,
) -> plt.Figure:
    """3-panel figure: raw MRI | Ground Truth overlay | Prediction overlay.

    Designed for the single-slice axial training mode. The background for all
    panels is the T1c modality (channel 1 by default), which best highlights
    the enhancing tumour. Overlaid colours:
        Green  — ET  (Enhancing Tumour, BraTS label 3)
        Red    — TC / NCR core (BraTS label 1)
        Yellow — WT / ED rim  (BraTS label 2)

    Args:
        image_slice:  float32 ndarray (4, H, W) — 4-modality axial MRI slice.
        gt_seg_slice: int32 ndarray (H, W) with BraTS labels, or None if no GT.
        pred_brats:   int32 ndarray (H, W) — predicted BraTS label map
                      (reconstructed from ET/TC/WT binary masks).
        case_id:      Subject identifier string (used in figure title).
        slice_idx:    Axial slice index (used in figure title).
        save_path:    If provided, save the PNG to this path.
        modality_idx: Which MRI channel to use as background (default 1 = T1c).

    Returns:
        Matplotlib Figure.
    """
    from matplotlib.patches import Patch

    bg = image_slice[modality_idx]          # (H, W) — T1c background
    H, W = bg.shape

    n_panels = 3 if gt_seg_slice is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    fig.patch.set_facecolor("#111111")

    modality_label = MODALITY_NAMES[modality_idx]
    title = (
        f"{case_id}  |  Axial slice {slice_idx}  |  Background: {modality_label}"
    )
    fig.suptitle(title, color="white", fontsize=12, y=1.02)

    def _show_panel(ax, seg_2d, panel_title):
        # Transpose because imshow with origin='lower' expects (cols, rows)
        ax.imshow(bg.T, cmap="gray", origin="lower", vmin=0, vmax=1)
        if seg_2d is not None:
            overlay = _seg_to_rgba(seg_2d.T, (W, H))
            ax.imshow(overlay, origin="lower")
        ax.set_title(panel_title, color="white", fontsize=10, pad=6)
        ax.axis("off")
        ax.set_facecolor("#111111")

    _show_panel(axes[0], None, "Raw MRI (T1c)")

    if gt_seg_slice is not None:
        _show_panel(axes[1], gt_seg_slice, "Ground Truth")
        _show_panel(axes[2], pred_brats,   "Prediction")
    else:
        _show_panel(axes[1], pred_brats, "Prediction")

    # Legend
    legend_elements = [
        Patch(facecolor=_OVERLAY_COLORS[3][:3], alpha=_OVERLAY_COLORS[3][3],
              label="ET  (Enhancing)"),
        Patch(facecolor=_OVERLAY_COLORS[1][:3], alpha=_OVERLAY_COLORS[1][3],
              label="TC / NCR Core"),
        Patch(facecolor=_OVERLAY_COLORS[2][:3], alpha=_OVERLAY_COLORS[2][3],
              label="WT / ED Rim"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=3,
        fontsize=9,
        framealpha=0.3,
        facecolor="#222222",
        labelcolor="white",
        bbox_to_anchor=(0.5, -0.06),
    )

    plt.tight_layout()
    if save_path is not None:
        save_path = _Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            str(save_path), dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
        plt.close(fig)

    return fig
