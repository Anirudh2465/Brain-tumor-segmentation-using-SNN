"""
ensemble.py — Multi-view ensemble fusion for 3D segmentation.

Procedure (Section 3.3 of arXiv:2601.16652v1):
  1. Run predict_volume() for each of the 3 view models on the same subject.
  2. Each returns a probability volume (3, H, W, D) in canonical orientation.
  3. Voxel-wise average across the 3 views → fused probability volume.
  4. Threshold at 0.5 → binary segmentation masks.

The paper reports ~44–48% NLL improvement from ensemble vs. single-view.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from spiking_useg.inference.predict import predict_volume
from spiking_useg.models.spiking_unet import SpikingUSegNet

VIEWS = ("sagittal", "coronal", "axial")


def load_model(
    checkpoint_path: Path | str,
    device: torch.device,
    model_kwargs: Optional[dict] = None,
) -> SpikingUSegNet:
    """Load a trained SpikingUSegNet from a checkpoint.

    Args:
        checkpoint_path: Path to best_model.pt saved by train_loop.py.
        device: Target device.
        model_kwargs: Dict of model constructor arguments (uses defaults if None).

    Returns:
        Loaded SpikingUSegNet model in eval mode.
    """
    if model_kwargs is None:
        model_kwargs = {}
    model = SpikingUSegNet(**model_kwargs)
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def ensemble_predict(
    view_checkpoints: dict[str, Path | str],
    data: np.ndarray,
    device: torch.device,
    model_kwargs: Optional[dict] = None,
    threshold: float = 0.5,
) -> dict[str, np.ndarray]:
    """Run multi-view ensemble inference on one subject.

    Args:
        view_checkpoints: Dict mapping view name → checkpoint path.
                          E.g. {"axial": "experiments/.../best_model.pt", ...}
        data: Preprocessed volume, float32, shape (4, 160, 192, 152).
        device: Inference device.
        model_kwargs: Constructor kwargs for SpikingUSegNet.
        threshold: Binarisation threshold for final segmentation.

    Returns:
        Dict with keys:
            'prob_ET', 'prob_TC', 'prob_WT' : float32 (160, 192, 152) — averaged probs
            'seg_ET', 'seg_TC', 'seg_WT'    : bool   (160, 192, 152) — binary masks
            'seg_brats'                      : int32  (160, 192, 152) — BraTS label map
                                               (reconstructed from ET/TC/WT masks)
    """
    per_view_probs: list[np.ndarray] = []

    for view, ckpt_path in view_checkpoints.items():
        model = load_model(ckpt_path, device, model_kwargs)
        prob_vol = predict_volume(model, data, view, device)  # (3, H, W, D)
        per_view_probs.append(prob_vol)
        del model  # free GPU memory

    # Average across views: (3, H, W, D)
    avg_probs = np.mean(per_view_probs, axis=0)

    et_prob = avg_probs[0]  # (H, W, D)
    tc_prob = avg_probs[1]
    wt_prob = avg_probs[2]

    et_seg = et_prob > threshold
    tc_seg = tc_prob > threshold
    wt_seg = wt_prob > threshold

    # Reconstruct approximate BraTS label map from hierarchical regions:
    #   WT = all tumor → 2 (ED) baseline
    #   TC ⊂ WT → set TC voxels to 1 (NCR)
    #   ET ⊂ TC → set ET voxels to 3 (ET)
    seg_brats = np.zeros(et_seg.shape, dtype=np.int32)
    seg_brats[wt_seg] = 2   # ED
    seg_brats[tc_seg] = 1   # NCR (overwrites ED inside TC)
    seg_brats[et_seg] = 3   # ET  (overwrites NCR inside ET)

    return {
        "prob_ET": et_prob,
        "prob_TC": tc_prob,
        "prob_WT": wt_prob,
        "seg_ET": et_seg,
        "seg_TC": tc_seg,
        "seg_WT": wt_seg,
        "seg_brats": seg_brats,
    }
