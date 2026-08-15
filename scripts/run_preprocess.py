"""
run_preprocess.py — CLI: Extract BraTS ZIPs, crop, normalize, create splits.

Usage:
    python scripts/run_preprocess.py [options]

Options:
    --data_dir      Path to data/ directory containing ZIP files (default: data)
    --subset        Number of subjects to process (0 = all, default: 0)
    --overwrite     Re-process even if output exists
    --n_folds       Number of CV folds (default: 5)
    --seed          Random seed for splits (default: 42)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running as script without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spiking_useg.data.preprocess import extract_zip, preprocess_dataset
from spiking_useg.data.splits import generate_splits, discover_case_ids
from spiking_useg.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BraTS 2023 preprocessing pipeline")
    p.add_argument("--data_dir", type=Path, default=Path("data"),
                   help="Directory containing BraTS ZIP files")
    p.add_argument("--subset", type=int, default=0,
                   help="Process only first N subjects (0 = all)")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-process even if output exists")
    p.add_argument("--n_folds", type=int, default=5,
                   help="Number of CV folds for splits")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for split generation")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)

    data_dir = args.data_dir.resolve()
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    splits_dir = data_dir / "splits"

    # ── Step 1: Extract training ZIP ──────────────────────────────────────
    train_zip = data_dir / "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip"
    if train_zip.exists():
        extract_zip(train_zip, raw_dir / "training")
    else:
        logger.warning("Training ZIP not found at %s - skipping extraction.", train_zip)

    # -- Step 2: Preprocess ------------------------------------------------
    logger.info("Starting preprocessing -> %s", processed_dir)
    processed_ids = preprocess_dataset(
        raw_dir=raw_dir / "training",
        out_dir=processed_dir,
        subset=args.subset,
        overwrite=args.overwrite,
    )
    logger.info("Preprocessed %d cases.", len(processed_ids))

    # ── Step 3: Discover all processed cases and generate splits ──────────
    all_ids = discover_case_ids(processed_dir)
    if not all_ids:
        logger.error("No processed cases found in %s.", processed_dir)
        sys.exit(1)

    logger.info("Generating %d-fold splits from %d cases ...", args.n_folds, len(all_ids))
    splits = generate_splits(
        case_ids=all_ids,
        n_folds=args.n_folds,
        splits_dir=splits_dir,
        dataset_name="brats23",
        seed=args.seed,
    )
    logger.info("Splits saved to %s", splits_dir)

    # Summary
    logger.info("-" * 60)
    logger.info("Preprocessing complete.")
    logger.info("  Processed dir : %s", processed_dir)
    logger.info("  Cases         : %d", len(all_ids))
    logger.info("  Folds         : %d", args.n_folds)
    for i, s in enumerate(splits):
        logger.info("  Fold %d: %d train, %d val", i, len(s["train"]), len(s["val"]))


if __name__ == "__main__":
    main()
