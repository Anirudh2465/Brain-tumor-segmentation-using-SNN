"""
train_all_folds.py — CLI: Train all views × all folds sequentially.

Launches up to 3 views × 5 folds = 15 training runs for BraTS 2023.

Usage:
    python scripts/train_all_folds.py [options]
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train all views × all folds for BraTS 2023")
    p.add_argument("--views", nargs="+", default=["axial", "coronal", "sagittal"])
    p.add_argument("--subset", type=int, default=0)
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--experiments_dir", type=str, default="experiments")
    p.add_argument("--dataset", type=str, default="brats23")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--fptt_alpha", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger(__name__)

    script = Path(__file__).parent / "train_single_view.py"
    total_runs = len(args.views)
    run_idx = 0
    fold = 0

    for view in args.views:
        run_idx += 1
        logger.info("-" * 60)
        logger.info("Run %d/%d - view=%s, fold=%d", run_idx, total_runs, view, fold)

        cmd = [
            sys.executable, str(script),
            "--view", view,
            "--fold", str(fold),
            "--data_dir", str(args.data_dir),
            "--experiments_dir", str(args.experiments_dir),
            "--dataset", args.dataset,
            "--max_subjects", str(args.subset),
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--fptt_alpha", str(args.fptt_alpha),
            "--patience", str(args.patience),
            "--seed", str(args.seed),
        ]

        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            logger.error("Run failed: view=%s fold=%d (exit code %d)", view, fold, result.returncode)

    logger.info("All %d runs complete.", total_runs)


if __name__ == "__main__":
    main()
