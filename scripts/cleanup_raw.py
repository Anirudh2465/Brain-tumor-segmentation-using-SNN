"""
cleanup_raw.py — Delete raw NIfTI case folders that have already been processed.

Use this after a run of run_preprocess.py that was NOT started with --delete_raw,
to reclaim the disk space used by the extracted NIfTI files.

For each case folder in raw_dir, checks whether data.npy exists in processed_dir.
If it does, the raw folder is deleted.  If not (processing failed or was skipped),
the raw folder is kept so you can re-process it later.

Usage:
    # Dry run — shows what WOULD be deleted without actually deleting:
    python scripts/cleanup_raw.py --dry_run

    # Actually delete:
    python scripts/cleanup_raw.py

    # Custom dirs:
    python scripts/cleanup_raw.py --raw_dir data/raw/training --processed_dir data/processed
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from spiking_useg.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Delete raw NIfTI folders that have already been preprocessed."
    )
    p.add_argument(
        "--raw_dir",
        type=Path,
        default=Path("data/raw/training"),
        help="Directory containing extracted BraTS NIfTI case folders.",
    )
    p.add_argument(
        "--processed_dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing processed .npy case folders.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be deleted without actually deleting anything.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)

    raw_dir       = args.raw_dir.resolve()
    processed_dir = args.processed_dir.resolve()

    if not raw_dir.exists():
        logger.error("Raw directory not found: %s", raw_dir)
        sys.exit(1)
    if not processed_dir.exists():
        logger.error("Processed directory not found: %s", processed_dir)
        sys.exit(1)

    if args.dry_run:
        logger.info("DRY RUN — nothing will be deleted.")

    # Find all case dirs in raw_dir (BraTS-GLI-* pattern)
    raw_case_dirs = sorted(
        p for p in raw_dir.iterdir()
        if p.is_dir() and p.name.startswith("BraTS-")
    )
    logger.info("Found %d raw case folders in %s", len(raw_case_dirs), raw_dir)

    n_deleted   = 0
    n_kept      = 0
    freed_bytes = 0

    for raw_case in raw_case_dirs:
        case_id      = raw_case.name
        proc_data    = processed_dir / case_id / "data.npy"

        if proc_data.exists():
            # Processed — safe to delete raw folder
            folder_size = sum(
                f.stat().st_size for f in raw_case.rglob("*") if f.is_file()
            )
            if args.dry_run:
                logger.info(
                    "  [DRY] Would delete: %s  (%.1f MB)",
                    raw_case.name, folder_size / 1e6,
                )
            else:
                shutil.rmtree(raw_case)
                logger.info(
                    "  Deleted: %s  (freed %.1f MB)", raw_case.name, folder_size / 1e6
                )
            freed_bytes += folder_size
            n_deleted   += 1
        else:
            # Not yet processed — keep it
            logger.info("  Kept (not yet processed): %s", raw_case.name)
            n_kept += 1

    action = "Would free" if args.dry_run else "Freed"
    logger.info("-" * 60)
    logger.info(
        "Done. %s %d case folders (%s %.1f GB).  %d kept (unprocessed).",
        "Would delete" if args.dry_run else "Deleted",
        n_deleted,
        action,
        freed_bytes / 1e9,
        n_kept,
    )

    if args.dry_run and n_deleted > 0:
        logger.info("Re-run without --dry_run to actually delete.")


if __name__ == "__main__":
    main()
