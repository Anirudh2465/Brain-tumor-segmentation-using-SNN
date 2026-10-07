"""
slice_dataset.py — Flat single-slice Dataset for axial-only training.

Instead of returning full (T, 4, H, W) sequences (as BraTSSliceDataset does),
this dataset returns ONE 2D axial slice per item:

    image  : float32 (4, H, W)  — one axial slice, 4 MRI modalities
    target : float32 (3, H, W)  — ET / TC / WT binary masks for that slice
    case_id: str
    slice_idx: int               — which axial slice index this is

The SNN temporal axis is collapsed to T=1 — model.reset_states() is called
before every forward pass in the training loop.

Empty-slice filtering:
    Most axial slices contain NO tumour at all (~60-70% of all slices across
    the dataset). Training on these blank slices wastes compute and hurts the
    loss signal.  At __init__ time we compute the tumour pixel fraction for
    every slice and only retain slices that exceed `min_tumor_fraction`.

Axial slice dimensions (after central crop 160×192×152):
    T = 152 slices, each (4, 160, 192)  →  H=160, W=192
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# Axial view config (axis=2 → T=152 slices, each 160×192)
_AXIAL_AXIS = 2
_AXIAL_T    = 152
_AXIAL_H    = 160
_AXIAL_W    = 192


# ── Augmentations (2D, no T dimension) ────────────────────────────────────────

class _RandomHFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(
        self, image: np.ndarray, target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if np.random.rand() < self.p:
            image  = image[:, :, ::-1].copy()   # flip W axis
            target = target[:, :, ::-1].copy()
        return image, target


class _RandomVFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(
        self, image: np.ndarray, target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if np.random.rand() < self.p:
            image  = image[:, ::-1, :].copy()   # flip H axis
            target = target[:, ::-1, :].copy()
        return image, target


class _IntensityJitter:
    def __init__(self, std: float = 0.01) -> None:
        self.std = std

    def __call__(
        self, image: np.ndarray, target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        noise = np.random.normal(0.0, self.std, size=image.shape).astype(np.float32)
        image = np.clip(image + noise, 0.0, 1.0)
        return image, target


class _Compose:
    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(
        self, image: np.ndarray, target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


def default_slice_augmentation() -> _Compose:
    """Standard 2D augmentation pipeline for single-slice training."""
    return _Compose([
        _RandomHFlip(p=0.5),
        _RandomVFlip(p=0.5),
        _IntensityJitter(std=0.01),
    ])


# ── Dataset ───────────────────────────────────────────────────────────────────

def _derive_binary_targets_2d(seg_slice: np.ndarray) -> np.ndarray:
    """Convert a single 2D BraTS label map to 3-channel binary targets.

    BraTS labels:  0=BG, 1=NCR, 2=ED, 3=ET
    Outputs (float32, shape (3, H, W)):
        [0] ET = label 3
        [1] TC = labels 1 + 3
        [2] WT = labels 1 + 2 + 3
    """
    et = (seg_slice == 3).astype(np.float32)
    tc = ((seg_slice == 1) | (seg_slice == 3)).astype(np.float32)
    wt = (seg_slice > 0).astype(np.float32)
    return np.stack([et, tc, wt], axis=0)  # (3, H, W)


class BraTSSingleSliceDataset(Dataset):
    """Flat dataset of individual axial slices from preprocessed BraTS volumes.

    Args:
        processed_dir: Root directory of per-case subdirectories
            (each containing data.npy and seg.npy).
        case_ids: List of case IDs to include.
        augment: If True, apply default 2D augmentation (training mode).
        min_tumor_fraction: Minimum fraction of tumour pixels to keep a slice.
            Slices below this threshold are discarded as background.
            Default 0.001 (0.1%) keeps slices with at least ~30 tumour pixels
            on a 160×192 slice.
        require_seg: Skip cases without seg.npy.
    """

    def __init__(
        self,
        processed_dir: Path | str,
        case_ids: list[str],
        augment: bool = False,
        min_tumor_fraction: float = 0.001,
        require_seg: bool = True,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.augment_fn = default_slice_augmentation() if augment else None
        self.min_tumor_fraction = min_tumor_fraction

        # ── Build flat index: list of (case_id, slice_idx) ────────────────
        self._index: list[tuple[str, int]] = []

        n_cases_skipped = 0
        for cid in case_ids:
            case_dir = self.processed_dir / cid
            data_path = case_dir / "data.npy"
            seg_path  = case_dir / "seg.npy"

            if not data_path.exists():
                logger.warning("data.npy missing for %s — skipping.", cid)
                n_cases_skipped += 1
                continue
            if require_seg and not seg_path.exists():
                logger.warning("seg.npy missing for %s — skipping.", cid)
                n_cases_skipped += 1
                continue

            if seg_path.exists():
                # Load seg to filter out empty slices
                seg = np.load(str(seg_path), mmap_mode="r")  # (H, W, D)
                # Axial slices → move axis 2 to front: (D, H, W)
                seg_axial = np.moveaxis(seg, _AXIAL_AXIS, 0)  # (152, 160, 192)
                total_pixels = seg_axial.shape[1] * seg_axial.shape[2]

                max_tumor_frac = -1.0
                best_t = -1
                for t in range(_AXIAL_T):
                    tumor_frac = float((seg_axial[t] > 0).sum()) / total_pixels
                    if tumor_frac > max_tumor_frac:
                        max_tumor_frac = tumor_frac
                        best_t = t
                        
                if max_tumor_frac >= min_tumor_fraction and best_t != -1:
                    self._index.append((cid, best_t))
            else:
                # No seg (inference mode) — just pick the middle slice
                self._index.append((cid, _AXIAL_T // 2))

        logger.info(
            "BraTSSingleSliceDataset: %d cases, %d tumour-containing slices "
            "(%.1f%% of all slices, %d cases skipped).",
            len(case_ids) - n_cases_skipped,
            len(self._index),
            100.0 * len(self._index) / max((len(case_ids) - n_cases_skipped) * _AXIAL_T, 1),
            n_cases_skipped,
        )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        """Return one axial slice as a training sample.

        Returns:
            dict with keys:
                'image'     : float32 tensor (4, 160, 192)
                'target'    : float32 tensor (3, 160, 192)
                'case_id'   : str
                'slice_idx' : int
        """
        case_id, slice_t = self._index[idx]
        case_dir = self.processed_dir / case_id

        # Load full volume (mmap — only the required slice is read from disk)
        data = np.load(str(case_dir / "data.npy"), mmap_mode="r")  # (4, H, W, D)
        # Extract axial slice: axis 2+1=3 in (C,H,W,D) → select index slice_t along D
        image = data[:, :, :, slice_t].copy()  # (4, 160, 192) — copy off mmap

        seg_path = case_dir / "seg.npy"
        if seg_path.exists():
            seg = np.load(str(seg_path), mmap_mode="r")   # (H, W, D)
            seg_slice = seg[:, :, slice_t].copy()           # (160, 192)
            target = _derive_binary_targets_2d(seg_slice)   # (3, 160, 192)
        else:
            target = None

        # Augmentation (2D, training only)
        if self.augment_fn is not None and target is not None:
            image, target = self.augment_fn(image, target)

        result: dict = {
            "image":     torch.from_numpy(image.astype(np.float32)),   # (4, H, W)
            "case_id":   case_id,
            "slice_idx": slice_t,
        }
        if target is not None:
            result["target"] = torch.from_numpy(target)   # (3, H, W)

        return result


# ── DataLoader factory ────────────────────────────────────────────────────────

def build_single_slice_dataloader(
    processed_dir: Path | str,
    case_ids: list[str],
    batch_size: int = 16,
    shuffle: bool = True,
    augment: bool = False,
    min_tumor_fraction: float = 0.001,
    require_seg: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Build a DataLoader for single-slice axial training.

    Args:
        processed_dir: Root processed data directory.
        case_ids: Subject IDs to include.
        batch_size: Slices per batch.
        shuffle: Shuffle between epochs.
        augment: Apply training augmentations.
        min_tumor_fraction: Filter threshold for empty slices.
        require_seg: Skip cases without segmentation masks.
        num_workers: DataLoader workers.

    Returns:
        PyTorch DataLoader yielding batches of shape (B, 4, H, W).
    """
    dataset = BraTSSingleSliceDataset(
        processed_dir=processed_dir,
        case_ids=case_ids,
        augment=augment,
        min_tumor_fraction=min_tumor_fraction,
        require_seg=require_seg,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
