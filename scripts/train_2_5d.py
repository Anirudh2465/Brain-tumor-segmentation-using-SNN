"""
train_2_5d.py — CLI to train Spiking U-Seg-Net in 2.5D Causal Spatiotemporal Mode.

Usage:
    python scripts/train_2_5d.py [options]

Options:
    --dataset         Dataset name (default: brats23)
    --fold            Fold number to train on (default: 0)
    --epochs          Number of training epochs (default: 50)
    --batch_size      Batch size (default: 16)
    --lr              Learning rate (default: 1e-3)
    --max_subjects    Max subjects to train on (0 = all, default: 0)
"""

from __future__ import annotations

import os
# Suppress TensorFlow C++ and oneDNN warnings triggered by TensorBoard
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import logging
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from spiking_useg.data.dataset import load_split
from spiking_useg.data.dataset_2_5d import build_2_5d_dataloader
from spiking_useg.models.spiking_unet import SpikingUSegNet
from spiking_useg.training.train_2_5d import train_2_5d_loop
from spiking_useg.training.scheduler import build_scheduler
from spiking_useg.utils.logging import setup_logging
from spiking_useg.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Spiking U-Seg-Net in 2.5D mode")
    p.add_argument("--data_dir", type=Path, default=Path("data"))
    p.add_argument("--experiments_dir", type=Path, default=Path("experiments"))
    p.add_argument("--dataset", type=str, default="brats23")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--min_tumor_frac", type=float, default=0.001)
    p.add_argument("--max_subjects", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--no_augment", action="store_true")
    p.add_argument("--no_tensorboard", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sequence_length", type=int, default=3,
                   help="Number of slices to pass in sequence (T)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    run_dir = (
        args.experiments_dir
        / args.dataset
        / f"fold{args.fold}"
        / "2_5d_attention"
    )
    setup_logging(log_dir=run_dir, run_name="train_2_5d", level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("  Spiking U-Seg-Net — 2.5D Causal Mode + Spatial Attention")
    logger.info("  Sequence Length (T) = %d slices", args.sequence_length)
    logger.info("  Dataset=%s  Fold=%d", args.dataset, args.fold)
    logger.info("=" * 60)

    splits_dir = args.data_dir / "splits"
    split = load_split(splits_dir, args.dataset, args.fold)
    train_ids: list[str] = split["train"]
    val_ids: list[str] = split["val"]

    if args.max_subjects > 0:
        train_ids = train_ids[: args.max_subjects]
        val_ids = val_ids[: max(1, args.max_subjects // 5)]
        logger.info("DEBUG MODE: %d train / %d val subjects.", len(train_ids), len(val_ids))

    processed_dir = args.data_dir / "processed"

    if not processed_dir.exists() or not any(processed_dir.iterdir()):
        logger.error("PREPROCESSED DATA NOT FOUND in %s", processed_dir)
        sys.exit(1)

    train_loader = build_2_5d_dataloader(
        processed_dir=processed_dir,
        case_ids=train_ids,
        batch_size=args.batch_size,
        shuffle=True,
        min_tumor_fraction=args.min_tumor_frac,
        require_seg=True,
        num_workers=args.num_workers,
        sequence_length=args.sequence_length,
    )
    val_loader = build_2_5d_dataloader(
        processed_dir=processed_dir,
        case_ids=val_ids,
        batch_size=max(1, args.batch_size // 2),
        shuffle=False,
        min_tumor_fraction=args.min_tumor_frac,
        require_seg=True,
        num_workers=args.num_workers,
        sequence_length=args.sequence_length,
    )

    if len(train_loader.dataset) == 0:
        logger.error("Training dataset is empty!")
        sys.exit(1)

    logger.info(
        "DataLoaders ready: %d train sequences / %d val sequences.",
        len(train_loader.dataset),
        len(val_loader.dataset),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpikingUSegNet()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = build_scheduler(optimizer, patience=2, factor=0.5)

    logger.info("Starting training ...")
    results = train_2_5d_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        run_dir=run_dir,
        use_tensorboard=not args.no_tensorboard,
    )

    logger.info("Training complete. Best Validation Dice: %.4f", results["best_val_dice"])


if __name__ == "__main__":
    main()
