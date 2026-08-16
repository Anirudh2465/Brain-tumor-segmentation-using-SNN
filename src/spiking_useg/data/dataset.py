"""
dataset.py — PyTorch Dataset and DataLoader for slice-sequence training.

Each item returned is one (subject × view) sequence:
    images : float32 tensor (T, 4, H, W) — ordered slice stack
    targets: float32 tensor (T, 3, H, W) — per-slice binary masks [ET, TC, WT]
    case_id: str — subject identifier (for logging/debugging)

The "T" dimension is the SNN temporal axis — each time step = one 2D slice.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from spiking_useg.data.slicing import (
    ViewName,
    derive_binary_targets,
    slice_segmentation,
    slice_volume,
)

logger = logging.getLogger(__name__)


class BraTSSliceDataset(Dataset):
    """Dataset of (subject × view) slice sequences from preprocessed BraTS data.

    Args:
        processed_dir: Root directory containing per-case subdirectories
            (each with data.npy and optionally seg.npy).
        case_ids: List of case IDs to include (e.g. ["BraTS-GLI-00000-000", ...]).
        view: Anatomical view — 'sagittal', 'coronal', or 'axial'.
        augment: Optional augmentation callable applied to (data_seq, seg_seq) pairs.
        require_seg: If True, skip cases without seg.npy (use for training splits).
    """

    def __init__(
        self,
        processed_dir: Path | str,
        case_ids: list[str],
        view: ViewName,
        augment=None,
        require_seg: bool = True,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.view = view
        self.augment = augment
        self.require_seg = require_seg

        # Validate and filter case IDs
        self.case_ids: list[str] = []
        for cid in case_ids:
            case_dir = self.processed_dir / cid
            if not (case_dir / "data.npy").exists():
                logger.warning("data.npy not found for case %s — skipping.", cid)
                continue
            if require_seg and not (case_dir / "seg.npy").exists():
                logger.warning("seg.npy not found for case %s — skipping.", cid)
                continue
            self.case_ids.append(cid)

        logger.info(
            "BraTSSliceDataset(%s, view=%s): %d cases loaded.",
            self.processed_dir.name, view, len(self.case_ids),
        )

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int) -> dict:
        """Return one (subject × view) sequence.

        Returns:
            dict with keys:
                'images'  : float32 tensor (T, 4, H, W)
                'targets' : float32 tensor (T, 3, H, W)  — if seg available
                'case_id' : str
        """
        case_id = self.case_ids[idx]
        case_dir = self.processed_dir / case_id

        # Load preprocessed volume (C, H, W, D)
        data = np.load(str(case_dir / "data.npy"), mmap_mode="r")  # (4,160,192,152)

        # Slice into sequence (T, 4, sH, sW)
        images = slice_volume(data, self.view)

        # Load and slice segmentation (T, sH, sW) → binary targets (T, 3, sH, sW)
        seg_path = case_dir / "seg.npy"
        if seg_path.exists():
            seg = np.load(str(seg_path), mmap_mode="r")  # (H, W, D)
            seg_seq = slice_segmentation(seg, self.view)  # (T, sH, sW)
            # Derive binary targets vectorised: (3, T, sH, sW) -> (T, 3, sH, sW)
            targets = derive_binary_targets(seg_seq).swapaxes(0, 1)
        else:
            # Validation set — no ground truth
            targets = None

        # Apply augmentation (operates on numpy arrays)
        if self.augment is not None and targets is not None:
            images, targets = self.augment(images, targets)

        result: dict = {
            "images": torch.from_numpy(images),  # (T, 4, H, W)
            "case_id": case_id,
        }
        if targets is not None:
            result["targets"] = torch.from_numpy(targets)  # (T, 3, H, W)

        return result


def build_dataloader(
    processed_dir: Path | str,
    case_ids: list[str],
    view: ViewName,
    batch_size: int = 8,
    shuffle: bool = True,
    augment=None,
    require_seg: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Convenience factory for building a DataLoader.

    Note: batch_size here means number of subject-sequences per batch.
    Each sequence has shape (T, 4, H, W). The DataLoader will stack them
    into (B, T, 4, H, W) if all T are equal (same view → all T are identical).

    Args:
        processed_dir: Root processed data directory.
        case_ids: Subject IDs to include.
        view: Anatomical view.
        batch_size: Number of sequences per batch.
        shuffle: Whether to shuffle between epochs.
        augment: Optional augmentation callable.
        require_seg: Skip cases without segmentation masks.
        num_workers: DataLoader workers (0 = main process; safe on Windows).

    Returns:
        PyTorch DataLoader.
    """
    dataset = BraTSSliceDataset(
        processed_dir=processed_dir,
        case_ids=case_ids,
        view=view,
        augment=augment,
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


def load_split(splits_dir: Path | str, dataset: str, fold: int) -> dict[str, list[str]]:
    """Load a fold's train/val case ID lists from JSON.

    Args:
        splits_dir: Directory containing split JSON files.
        dataset: Dataset name (e.g. 'brats23').
        fold: Fold index (0-based).

    Returns:
        dict with 'train' and 'val' keys, each a list of case ID strings.
    """
    split_path = Path(splits_dir) / f"{dataset}_fold{fold}.json"
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    with open(split_path, "r") as f:
        return json.load(f)
