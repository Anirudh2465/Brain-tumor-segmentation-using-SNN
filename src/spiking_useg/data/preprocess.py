"""
preprocess.py — BraTS 2023 preprocessing pipeline.

Steps (matching Section 4.1 of arXiv:2601.16652v1):
  1. Extract ZIP archives if needed.
  2. Confirm/assert 240×240×155 input dimensions.
  3. Central crop each volume from 240×240×155 → 160×192×152.
  4. Min–max normalize each modality per volume to [0, 1].
  5. Save processed volumes as .npy files under processed_dir/.

Output structure:
    processed_dir/
        <case_id>/
            data.npy      # float32, shape (4, 160, 192, 152) — (C, H, W, D)
            seg.npy       # int32,   shape (160, 192, 152)    — label map (train only)
"""

from __future__ import annotations

import logging
import os
import zipfile
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# BraTS modality suffixes in the desired channel order (matches configs/base.yaml)
MODALITY_SUFFIXES = ["t1n", "t1c", "t2w", "t2f"]

# Expected raw volume shape (H, W, D) from BraTS 2023
EXPECTED_RAW_SHAPE = (240, 240, 155)

# Target cropped shape (H, W, D)
CROP_SHAPE = (160, 192, 152)


def _central_crop(
    volume: np.ndarray, target_shape: tuple[int, int, int]
) -> np.ndarray:
    """Crop a 3D volume (H, W, D) to target_shape around the center.

    Args:
        volume: ndarray of shape (H, W, D).
        target_shape: Desired (H', W', D').

    Returns:
        Cropped ndarray of shape target_shape.
    """
    h, w, d = volume.shape
    th, tw, td = target_shape
    sh = (h - th) // 2
    sw = (w - tw) // 2
    sd = (d - td) // 2
    return volume[sh : sh + th, sw : sw + tw, sd : sd + td]


def _minmax_normalize(volume: np.ndarray) -> np.ndarray:
    """Min–max normalize a 3D array to [0, 1].

    Applied per-volume (whole volume), not per-slice.
    This matches common BraTS practice (deviation D8).

    Args:
        volume: 3D ndarray.

    Returns:
        Float32 ndarray in [0, 1].
    """
    vmin = float(volume.min())
    vmax = float(volume.max())
    if vmax - vmin < 1e-8:
        return np.zeros_like(volume, dtype=np.float32)
    return ((volume - vmin) / (vmax - vmin)).astype(np.float32)


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    """Extract a ZIP archive to out_dir if not already extracted.

    Args:
        zip_path: Path to the .zip file.
        out_dir: Destination directory.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # Heuristic: skip extraction if there are already subdirectories present
    existing = [p for p in out_dir.iterdir() if p.is_dir()]
    if existing:
        logger.info(
            "Skipping extraction of %s — %d directories already present in %s",
            zip_path.name,
            len(existing),
            out_dir,
        )
        return

    logger.info("Extracting %s → %s (this may take a while) …", zip_path.name, out_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        for member in tqdm(members, desc=f"Extracting {zip_path.name}", unit="file"):
            zf.extract(member, out_dir)
    logger.info("Extraction complete.")


def list_case_dirs(raw_dir: Path, prefix: str = "BraTS-GLI-") -> list[Path]:
    """Return sorted list of case directories under raw_dir.

    Args:
        raw_dir: Directory containing the extracted BraTS data. May contain
            a single sub-folder (the extracted ZIP root) that itself contains
            the case directories.
        prefix: Expected directory name prefix for case folders.

    Returns:
        Sorted list of Path objects, one per case.
    """
    candidates: list[Path] = []
    for p in raw_dir.rglob("*"):
        if p.is_dir() and p.name.startswith(prefix):
            candidates.append(p)
    return sorted(candidates)


def preprocess_case(
    case_dir: Path,
    out_dir: Path,
    crop_shape: tuple[int, int, int] = CROP_SHAPE,
    overwrite: bool = False,
) -> Optional[Path]:
    """Preprocess one BraTS case: crop + normalize + save.

    Args:
        case_dir: Path to the case directory containing NIfTI files.
        out_dir: Root output directory for processed data.
        crop_shape: Target (H, W, D) after central crop.
        overwrite: If True, reprocess even if output already exists.

    Returns:
        Path to the output case directory, or None if skipped.
    """
    case_id = case_dir.name
    out_case_dir = out_dir / case_id
    out_data_path = out_case_dir / "data.npy"

    if out_data_path.exists() and not overwrite:
        return out_case_dir  # Already processed

    out_case_dir.mkdir(parents=True, exist_ok=True)

    # ── Load modalities ────────────────────────────────────────────────────
    modality_volumes: list[np.ndarray] = []
    for suffix in MODALITY_SUFFIXES:
        nii_path = case_dir / f"{case_id}-{suffix}.nii.gz"
        if not nii_path.exists():
            logger.warning("Missing modality file: %s — skipping case %s", nii_path, case_id)
            return None
        img = nib.load(str(nii_path))
        vol = img.get_fdata(dtype=np.float32)  # (H, W, D) or (240, 240, 155)

        # Warn if shape is unexpected (but don't crash — some BraTS cases differ slightly)
        if vol.shape != EXPECTED_RAW_SHAPE:
            logger.warning(
                "Case %s modality %s has shape %s (expected %s)",
                case_id, suffix, vol.shape, EXPECTED_RAW_SHAPE,
            )

        # Crop
        vol = _central_crop(vol, crop_shape)
        # Normalize
        vol = _minmax_normalize(vol)
        modality_volumes.append(vol)

    # Stack to (4, H, W, D)
    data = np.stack(modality_volumes, axis=0)  # (C=4, H, W, D)
    np.save(str(out_data_path), data)

    # ── Load segmentation mask (training only) ─────────────────────────────
    seg_path = case_dir / f"{case_id}-seg.nii.gz"
    if seg_path.exists():
        seg_img = nib.load(str(seg_path))
        seg = seg_img.get_fdata().astype(np.int32)
        seg = _central_crop(seg, crop_shape)
        np.save(str(out_case_dir / "seg.npy"), seg)

    return out_case_dir


def preprocess_dataset(
    raw_dir: Path,
    out_dir: Path,
    subset: int = 0,
    overwrite: bool = False,
) -> list[str]:
    """Preprocess all (or a subset of) BraTS cases.

    Args:
        raw_dir: Directory containing extracted BraTS NIfTI case folders.
        out_dir: Root directory for processed output.
        subset: If > 0, process only the first `subset` cases (for debugging).
        overwrite: Reprocess even if output exists.

    Returns:
        List of processed case IDs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    case_dirs = list_case_dirs(raw_dir)

    if not case_dirs:
        raise FileNotFoundError(
            f"No BraTS case directories found under {raw_dir}. "
            "Did you forget to extract the ZIP file?"
        )

    if subset > 0:
        case_dirs = case_dirs[:subset]
        logger.info("Processing subset of %d cases.", subset)
    else:
        logger.info("Processing all %d cases.", len(case_dirs))

    processed_ids: list[str] = []
    for case_dir in tqdm(case_dirs, desc="Preprocessing cases", unit="case"):
        result = preprocess_case(case_dir, out_dir, overwrite=overwrite)
        if result is not None:
            processed_ids.append(case_dir.name)

    logger.info("Preprocessing done. %d/%d cases processed.", len(processed_ids), len(case_dirs))
    return processed_ids


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys
    # Quick standalone test: python -m spiking_useg.data.preprocess <raw_dir> <out_dir> [subset]
    raw = Path(sys.argv[1])
    out = Path(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    ids = preprocess_dataset(raw, out, subset=n)
    print(f"Processed {len(ids)} cases.")
