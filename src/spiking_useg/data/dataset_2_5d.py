"""
dataset_2_5d.py — 2.5D Causal Data Loader for Spiking U-Seg-Net.

Creates sequences of adjacent axial slices [Z-2, Z-1, Z] to be processed
causally by the SNN. The target mask is only provided for the final slice Z.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from spiking_useg.data.slice_dataset import _derive_binary_targets_2d

logger = logging.getLogger(__name__)

_AXIAL_AXIS = 2
_AXIAL_T = 152  # BraTS Z dimension after crop (160, 192, 152)

class BraTS25DDataset(Dataset):
    """Yields 2.5D causal sequences (T=3 slices) for each valid target slice.
    
    Sequence structure: [slice_Z-2, slice_Z-1, slice_Z].
    If Z-1 or Z-2 are out of bounds (< 0), the boundary slice is repeated.
    The ground truth target is returned only for the final slice Z.
    """

    def __init__(
        self,
        processed_dir: Path | str,
        case_ids: list[str],
        min_tumor_fraction: float = 0.001,
        require_seg: bool = True,
        sequence_length: int = 3,
    ) -> None:
        super().__init__()
        self.processed_dir = Path(processed_dir)
        self.require_seg = require_seg
        self.sequence_length = sequence_length
        self._index: list[tuple[str, int]] = []

        n_cases_skipped = 0

        for cid in case_ids:
            case_dir = self.processed_dir / cid
            if not (case_dir / "data.npy").exists():
                n_cases_skipped += 1
                continue

            seg_path = case_dir / "seg.npy"
            if self.require_seg:
                if not seg_path.exists():
                    n_cases_skipped += 1
                    continue

                seg = np.load(str(seg_path), mmap_mode="r")  # (160, 192, 152)
                seg_axial = np.moveaxis(seg, _AXIAL_AXIS, 0)
                total_pixels = seg_axial.shape[1] * seg_axial.shape[2]

                for t in range(_AXIAL_T):
                    tumor_frac = float((seg_axial[t] > 0).sum()) / total_pixels
                    if tumor_frac >= min_tumor_fraction:
                        # Add target slice Z
                        self._index.append((cid, t))
            else:
                for t in range(_AXIAL_T):
                    self._index.append((cid, t))

        logger.info(
            "BraTS25DDataset: %d cases, %d sequence targets "
            "(%.1f%% of all slices, %d cases skipped).",
            len(case_ids) - n_cases_skipped,
            len(self._index),
            100.0 * len(self._index) / max((len(case_ids) - n_cases_skipped) * _AXIAL_T, 1),
            n_cases_skipped,
        )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        case_id, slice_z = self._index[idx]
        case_dir = self.processed_dir / case_id

        data = np.load(str(case_dir / "data.npy"), mmap_mode="r")  # (4, H, W, D)
        
        # Build sequence of slices [Z-seq_len+1, ..., Z]
        seq_slices = []
        for offset in range(self.sequence_length - 1, -1, -1):
            t = slice_z - offset
            t = max(0, min(t, _AXIAL_T - 1))  # Clamp to valid range
            seq_slices.append(data[:, :, :, t].copy())

        # Stack into (T, 4, H, W)
        image_seq = np.stack(seq_slices, axis=0)

        seg_path = case_dir / "seg.npy"
        if seg_path.exists():
            seg = np.load(str(seg_path), mmap_mode="r")
            seg_slice_z = seg[:, :, slice_z].copy()
            target = _derive_binary_targets_2d(seg_slice_z)
        else:
            target = None

        result: dict = {
            "image":     torch.from_numpy(image_seq.astype(np.float32)),  # (T, 4, H, W)
            "case_id":   case_id,
            "slice_idx": slice_z,
        }
        if target is not None:
            result["target"] = torch.from_numpy(target)   # (3, H, W)

        return result


def build_2_5d_dataloader(
    processed_dir: Path | str,
    case_ids: list[str],
    batch_size: int = 16,
    shuffle: bool = True,
    min_tumor_fraction: float = 0.001,
    require_seg: bool = True,
    num_workers: int = 0,
    sequence_length: int = 3,
) -> DataLoader:
    dataset = BraTS25DDataset(
        processed_dir=processed_dir,
        case_ids=case_ids,
        min_tumor_fraction=min_tumor_fraction,
        require_seg=require_seg,
        sequence_length=sequence_length,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
