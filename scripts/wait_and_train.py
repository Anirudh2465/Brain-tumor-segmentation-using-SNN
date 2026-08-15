"""
wait_and_train.py - Waits for preprocessing to finish then starts training.

Polls data/processed until count matches 1251, then launches train_all_folds.py.
Run this in a separate terminal alongside run_preprocess.py:

    python scripts/wait_and_train.py
"""

from __future__ import annotations

import subprocess
import sys
import time
import logging
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROCESSED_DIR = ROOT / "data" / "processed"
TARGET_COUNT  = 1251   # total BraTS 2023 training subjects
POLL_INTERVAL = 30     # seconds between checks


def count_processed() -> int:
    if not PROCESSED_DIR.exists():
        return 0
    return sum(
        1 for p in PROCESSED_DIR.iterdir()
        if p.is_dir() and (p / "data.npy").exists()
    )


def main():
    logger.info("Waiting for preprocessing to finish (%d subjects)...", TARGET_COUNT)

    while True:
        done = count_processed()
        pct = done / TARGET_COUNT * 100
        logger.info("  Preprocessed: %d / %d (%.1f%%)", done, TARGET_COUNT, pct)

        if done >= TARGET_COUNT:
            logger.info("Preprocessing complete! Starting training...")
            break

        time.sleep(POLL_INTERVAL)

    # Launch training
    cmd = [
        sys.executable, str(ROOT / "scripts" / "run_full_pipeline.py"),
        "--skip_preprocess",     # preprocess already done
        "--epochs",  "100",
        "--batch_size", "8",
        "--fptt_alpha", "0.1",
        "--patience",   "10",
    ]
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    main()
