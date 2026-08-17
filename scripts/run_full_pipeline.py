"""
run_full_pipeline.py - Master orchestration script.

Runs:
  1. Preprocessing (all subjects, skip already done)
  2. Training all views x all folds
  3. Ensemble evaluation on each fold
  4. Final report (Dice + NLL tables)

Usage:
    python scripts/run_full_pipeline.py [--epochs 100] [--batch_size 8]
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run(cmd: list[str], desc: str) -> int:
    """Run a subprocess command, logging start/end. Returns exit code."""
    logger.info("=== START: %s ===", desc)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - t0
    if result.returncode == 0:
        logger.info("=== DONE: %s (%.1fs) ===", desc, elapsed)
    else:
        logger.error("=== FAILED: %s (exit %d) ===", desc, result.returncode)
    return result.returncode


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full Spiking U-Seg-Net pipeline")
    p.add_argument("--data_dir",        type=str, default="data")
    p.add_argument("--experiments_dir", type=str, default="experiments")
    p.add_argument("--views",   nargs="+", default=["axial", "coronal", "sagittal"])
    p.add_argument("--subset",      type=int,   default=0, help="Train on a subset of data (0 = all)")
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--batch_size",  type=int,   default=8)
    p.add_argument("--fptt_alpha",  type=float, default=0.1)
    p.add_argument("--patience",    type=int,   default=10)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--skip_preprocess",  action="store_true",
                   help="Skip preprocessing (data already prepared)")
    p.add_argument("--skip_training",    action="store_true",
                   help="Skip training (load existing checkpoints)")
    return p.parse_args()


def main():
    args = parse_args()
    py = sys.executable
    pythonpath_env = f"PYTHONPATH={ROOT / 'src'}"

    # ── Phase 1: Preprocessing ─────────────────────────────────────────────
    if not args.skip_preprocess:
        rc = run(
            [py, "scripts/run_preprocess.py",
             "--data_dir", args.data_dir,
             "--subset", str(args.subset),
             "--n_folds", "1",
             "--seed", str(args.seed)],
            f"Preprocessing ({args.subset if args.subset > 0 else 'all'} subjects)"
        )
        if rc != 0:
            logger.error("Preprocessing failed. Aborting.")
            sys.exit(rc)
    else:
        logger.info("Skipping preprocessing (--skip_preprocess)")

    # ── Phase 2: Training all views for fold 0 ─────────────────────────────
    if not args.skip_training:
        rc = run(
            [py, "scripts/train_all_views.py",
             "--views"] + args.views + [
             "--data_dir",        args.data_dir,
             "--experiments_dir", args.experiments_dir,
             "--subset",     str(args.subset),
             "--epochs",     str(args.epochs),
             "--batch_size", str(args.batch_size),
             "--fptt_alpha", str(args.fptt_alpha),
             "--patience",   str(args.patience),
             "--seed",       str(args.seed)],
            f"Training ({len(args.views)} views)"
        )
        if rc != 0:
            logger.warning("Training returned non-zero exit. Continuing to evaluation.")
    else:
        logger.info("Skipping training (--skip_training)")

    # ── Phase 3: Ensemble evaluation for fold 0 ────────────────────────────
    rc = run(
        [py, "scripts/run_ensemble_eval.py",
         "--fold",           "0",
         "--data_dir",       args.data_dir,
         "--experiments_dir", args.experiments_dir,
         "--views"] + args.views + [
         "--seed", str(args.seed)],
        "Ensemble evaluation fold 0"
    )
    if rc != 0:
        logger.warning("Ensemble eval fold 0 failed (exit %d). Continuing.", rc)

    # ── Phase 4: Print results tables ─────────────────────────────────────
    rc = run(
        [py, "-c",
         f"import sys; sys.path.insert(0,'src'); "
         f"from spiking_useg.eval.report import generate_dice_table, generate_nll_table, generate_flops_table; "
         f"generate_dice_table('{args.experiments_dir}','brats23'); "
         f"generate_nll_table('{args.experiments_dir}','brats23'); "
         f"generate_flops_table('{args.experiments_dir}','brats23')"],
        "Generating results tables"
    )

    logger.info("=" * 60)
    logger.info("Pipeline complete.")
    logger.info("  Experiments: %s", Path(args.experiments_dir).resolve())
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
