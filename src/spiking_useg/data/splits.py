"""
splits.py — Generate 5-fold subject-level cross-validation splits for BraTS 2023.

Output: data/splits/brats23_fold{k}.json for k in 0..4

Each JSON file has the structure:
    {
        "train": ["BraTS-GLI-00000-000", ...],
        "val":   ["BraTS-GLI-00251-000", ...]
    }

Split strategy (matching project plan, Section 4, Data Pipeline):
  - BraTS23: 1,251 subjects → fold sizes 251/250/250/250/250
  - Subjects with longitudinal follow-up (-001 sessions) are grouped so that
    all sessions of the same subject fall in the same split (avoids data leakage).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _group_by_subject(case_ids: list[str]) -> dict[str, list[str]]:
    """Group case IDs by their 5-digit subject ID to handle longitudinal scans.

    Args:
        case_ids: List like ["BraTS-GLI-00000-000", "BraTS-GLI-00000-001", ...]

    Returns:
        Dict mapping subject_id → list of case IDs.
    """
    groups: dict[str, list[str]] = {}
    for cid in sorted(case_ids):
        # Format: BraTS-GLI-{XXXXX}-{YYY}
        parts = cid.split("-")
        subject_id = "-".join(parts[:3])  # BraTS-GLI-XXXXX
        groups.setdefault(subject_id, []).append(cid)
    return groups


def generate_splits(
    case_ids: list[str],
    n_folds: int = 5,
    splits_dir: Path | str = "data/splits",
    dataset_name: str = "brats23",
    seed: int = 42,
) -> list[dict[str, list[str]]]:
    """Generate and save n-fold cross-validation splits.

    Args:
        case_ids: All available case IDs (from processed_dir listing).
        n_folds: Number of CV folds.
        splits_dir: Directory to save JSON split files.
        dataset_name: Prefix for JSON filenames.
        seed: Random seed for reproducibility.

    Returns:
        List of dicts (one per fold), each with 'train' and 'val' lists.
    """
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    # Group by subject to prevent longitudinal data leakage
    groups = _group_by_subject(case_ids)
    subject_ids = np.array(sorted(groups.keys()))
    rng.shuffle(subject_ids)

    # Split subject IDs into n_folds groups
    subject_folds = np.array_split(subject_ids, n_folds)

    fold_splits: list[dict[str, list[str]]] = []
    for fold_idx in range(n_folds):
        val_subjects = set(subject_folds[fold_idx].tolist())
        train_subjects = set(s for s in subject_ids if s not in val_subjects)

        val_cases: list[str] = []
        train_cases: list[str] = []
        for subject_id, cases in groups.items():
            if subject_id in val_subjects:
                val_cases.extend(cases)
            else:
                train_cases.extend(cases)

        val_cases.sort()
        train_cases.sort()

        split = {"train": train_cases, "val": val_cases}
        fold_splits.append(split)

        out_path = splits_dir / f"{dataset_name}_fold{fold_idx}.json"
        with open(out_path, "w") as f:
            json.dump(split, f, indent=2)

        logger.info(
            "Fold %d: %d train, %d val -> saved to %s",
            fold_idx, len(train_cases), len(val_cases), out_path,
        )

    return fold_splits


def discover_case_ids(processed_dir: Path | str) -> list[str]:
    """List all case IDs found in a processed data directory.

    Args:
        processed_dir: Directory containing per-case subdirectories.

    Returns:
        Sorted list of case IDs (directory names that contain data.npy).
    """
    processed_dir = Path(processed_dir)
    case_ids = [
        p.name
        for p in sorted(processed_dir.iterdir())
        if p.is_dir() and (p / "data.npy").exists()
    ]
    return case_ids
