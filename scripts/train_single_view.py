"""
train_single_view.py — CLI: Train one view × one fold.

Usage:
    python scripts/train_single_view.py [options]

Example (smoke test on 5 subjects, axial view, fold 0):
    python scripts/train_single_view.py \
        --view axial --fold 0 --max_subjects 5 --epochs 3

Example (full training):
    python scripts/train_single_view.py --view axial --fold 0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from spiking_useg.data.augment import default_train_augmentation
from spiking_useg.data.dataset import build_dataloader, load_split
from spiking_useg.models.spiking_unet import SpikingUSegNet
from spiking_useg.training.train_loop import train
from spiking_useg.utils.logging import setup_logging
from spiking_useg.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Spiking U-Seg-Net on one view/fold")
    p.add_argument("--view", choices=["axial", "coronal", "sagittal"], default="axial")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--data_dir", type=Path, default=Path("data"))
    p.add_argument("--experiments_dir", type=Path, default=Path("experiments"))
    p.add_argument("--dataset", type=str, default="brats23")
    p.add_argument("--max_subjects", type=int, default=0,
                   help="Limit subjects for debugging (0 = all)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--fptt_alpha", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=0.3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_augment", action="store_true")
    p.add_argument("--no_tensorboard", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    run_dir = args.experiments_dir / args.dataset / f"fold{args.fold}" / args.view
    setup_logging(log_dir=run_dir, run_name="train", level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("=== Training: dataset=%s  view=%s  fold=%d ===", args.dataset, args.view, args.fold)

    # ── Load split ─────────────────────────────────────────────────────────
    splits_dir = args.data_dir / "splits"
    split = load_split(splits_dir, args.dataset, args.fold)

    train_ids = split["train"]
    val_ids = split["val"]

    if args.max_subjects > 0:
        train_ids = train_ids[:args.max_subjects]
        val_ids = val_ids[:max(1, args.max_subjects // 5)]
        logger.info("DEBUG MODE: limited to %d train / %d val subjects.", len(train_ids), len(val_ids))

    processed_dir = args.data_dir / "processed"

    augment = None if args.no_augment else default_train_augmentation()

    train_loader = build_dataloader(
        processed_dir=processed_dir,
        case_ids=train_ids,
        view=args.view,
        batch_size=args.batch_size,
        shuffle=True,
        augment=augment,
        require_seg=True,
    )

    val_loader = build_dataloader(
        processed_dir=processed_dir,
        case_ids=val_ids,
        view=args.view,
        batch_size=1,
        shuffle=False,
        augment=None,
        require_seg=True,
    )

    # ── Build model ────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpikingUSegNet()
    
    if torch.cuda.device_count() > 1:
        logger.info("Using %d GPUs with DataParallel!", torch.cuda.device_count())
        model = torch.nn.DataParallel(model)
        
    model.to(device)
    logger.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))

    # ── Train ──────────────────────────────────────────────────────────────
    results = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        run_dir=run_dir,
        max_epochs=args.epochs,
        lr=args.lr,
        fptt_alpha=args.fptt_alpha,
        grad_clip_norm=args.grad_clip,
        patience=args.patience,
        device=device,
        use_tensorboard=not args.no_tensorboard,
    )

    logger.info("Final best val_dice: %.4f", results["best_val_dice"])

    # ── Measure FLOPs ──────────────────────────────────────────────────────
    from spiking_useg.efficiency.flops import measure_snn_flops
    import json
    
    # Reload best model weights for accurate spike rates
    best_model_path = run_dir / "best_model.pth"
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        
    logger.info("Measuring spike-rate-aware FLOPs on validation batch...")
    try:
        sample_shape = tuple(val_loader.dataset[0]["images"].shape[1:]) # (C, H, W)
        flops_dict = measure_snn_flops(
            model=model,
            input_shape=sample_shape,
            dataloader=val_loader,
            device=device
        )
        with open(run_dir / "flops.json", "w") as f:
            json.dump(flops_dict, f, indent=2)
        logger.info("FLOPs measured and saved to %s", run_dir / "flops.json")
    except Exception as e:
        logger.error("Failed to measure FLOPs: %s", e)


if __name__ == "__main__":
    main()
