"""
eval_single_slice.py — Evaluate a trained Spiking U-Seg-Net single-slice model.

Loads the best model checkpoint, evaluates it on the validation set to calculate
Dice metrics, and generates visualization overlays for the specified number of cases.

Usage:
    python scripts/eval_single_slice.py --fold 0 --vis_cases 5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from spiking_useg.data.slice_dataset import build_single_slice_dataloader
from spiking_useg.data.dataset import load_split
from spiking_useg.models.spiking_unet import SpikingUSegNet
from spiking_useg.training.train_slice import validate_one_epoch_slices
from spiking_useg.utils.logging import setup_logging
from spiking_useg.utils.seed import seed_everything
from train_single_slice import run_visualization


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate trained single-slice model.")
    p.add_argument("--data_dir", type=Path, default=Path("data"))
    p.add_argument("--experiments_dir", type=Path, default=Path("experiments"))
    p.add_argument("--dataset", type=str, default="brats23")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--vis_cases", type=int, default=5,
                   help="Number of cases to visualize (0 = none)")
    p.add_argument("--vis_only", action="store_true",
                   help="Skip metric evaluation and only generate visualizations")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    run_dir = args.experiments_dir / args.dataset / f"fold{args.fold}" / "axial_slice"
    setup_logging(log_dir=run_dir, run_name="eval_slice", level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("  Evaluating Single-Slice Axial Mode")
    logger.info("  Dataset=%s  Fold=%d", args.dataset, args.fold)
    logger.info("=" * 60)

    ckpt_path = run_dir / "best_model.pt"
    if not ckpt_path.exists():
        logger.error("No trained model found at %s", ckpt_path)
        sys.exit(1)

    # ── Load fold split ──────────────────────────────────────────────────
    splits_dir = args.data_dir / "splits"
    split = load_split(splits_dir, args.dataset, args.fold)
    val_ids: list[str] = split["val"]

    processed_dir = args.data_dir / "processed"

    val_loader = build_single_slice_dataloader(
        processed_dir=processed_dir,
        case_ids=val_ids,
        batch_size=args.batch_size,
        shuffle=False,
        augment=False,
        min_tumor_fraction=0.001,
        require_seg=True,
    )
    logger.info("Validation DataLoader ready: %d slices.", len(val_loader.dataset))

    # ── Build & load model ───────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpikingUSegNet()
    
    logger.info("Loading model from %s", ckpt_path)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    
    # Handle DataParallel if it was saved with it
    state_dict = ckpt["model_state_dict"]
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        
    model.to(device)
    model.eval()

    # ── Evaluation ───────────────────────────────────────────────────────
    if not args.vis_only:
        logger.info("Running evaluation on validation set...")
        metrics = validate_one_epoch_slices(model, val_loader, device)
        
        logger.info("Evaluation Results:")
        logger.info("  Mean Dice (Overall): %.4f", metrics["val_dice"])
        logger.info("  Enhancing Tumor (ET) Dice: %.4f", metrics["val_dice_ET"])
        logger.info("  Tumor Core (TC) Dice:      %.4f", metrics["val_dice_TC"])
        logger.info("  Whole Tumor (WT) Dice:     %.4f", metrics["val_dice_WT"])
    else:
        logger.info("Skipping metrics evaluation (--vis_only flag set).")

    # ── Visualization ────────────────────────────────────────────────────
    if args.vis_cases > 0:
        logger.info("Generating visualizations...")
        vis_dir = run_dir / "visualizations_eval"
        
        # Unwrap DataParallel for visualization if needed
        unwrapped_model = model.module if hasattr(model, "module") else model
        
        run_visualization(
            model=unwrapped_model,
            val_ids=val_ids,
            processed_dir=processed_dir,
            vis_dir=vis_dir,
            device=device,
            n_cases=args.vis_cases,
        )


if __name__ == "__main__":
    main()
