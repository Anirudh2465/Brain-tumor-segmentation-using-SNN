"""
eval_2_5d.py — Evaluate the 2.5D Causal Attention model and generate visualization grids.

Generates a 3x3 grid comparing Raw T1c, Predicted Subregion, and Actual Subregion
for ET, TC, and WT.
"""

from __future__ import annotations

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import logging
import sys
from pathlib import Path
import numpy as np

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from spiking_useg.data.dataset_2_5d import build_2_5d_dataloader, BraTS25DDataset
from spiking_useg.data.dataset import load_split
from spiking_useg.models.spiking_unet import SpikingUSegNet
from spiking_useg.training.train_2_5d import validate_one_epoch_2_5d
from spiking_useg.utils.logging import setup_logging
from spiking_useg.utils.seed import seed_everything
from spiking_useg.utils.viz import plot_subregion_comparison


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate trained 2.5D model.")
    p.add_argument("--data_dir", type=Path, default=Path("data"))
    p.add_argument("--experiments_dir", type=Path, default=Path("experiments"))
    p.add_argument("--dataset", type=str, default="brats23")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--vis_cases", type=int, default=5,
                   help="Number of cases to visualize (0 = none)")
    p.add_argument("--vis_only", action="store_true",
                   help="Skip metric evaluation and only generate visualizations")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    run_dir = args.experiments_dir / args.dataset / f"fold{args.fold}" / "2_5d_attention"
    setup_logging(log_dir=run_dir, run_name="eval_2_5d", level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("  Evaluating 2.5D Causal Mode + Spatial Attention")
    logger.info("  Dataset=%s  Fold=%d", args.dataset, args.fold)
    logger.info("=" * 60)

    ckpt_path = run_dir / "best_model.pt"
    if not ckpt_path.exists():
        logger.error("No trained model found at %s", ckpt_path)
        sys.exit(1)

    # Load fold split
    splits_dir = args.data_dir / "splits"
    split = load_split(splits_dir, args.dataset, args.fold)
    val_ids: list[str] = split["val"]

    processed_dir = args.data_dir / "processed"

    val_loader = build_2_5d_dataloader(
        processed_dir=processed_dir,
        case_ids=val_ids,
        batch_size=args.batch_size,
        shuffle=False,
        min_tumor_fraction=0.01,
        require_seg=True,
        num_workers=args.num_workers,
    )
    logger.info("Validation DataLoader ready: %d sequences.", len(val_loader.dataset))

    # Build & load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpikingUSegNet()
    
    logger.info("Loading model from %s", ckpt_path)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    
    state_dict = ckpt["model_state_dict"]
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Evaluation
    if not args.vis_only:
        logger.info("Running evaluation on validation set...")
        metrics = validate_one_epoch_2_5d(model, val_loader, device)
        
        logger.info("Evaluation Results:")
        logger.info("  Mean Dice (Overall): %.4f", metrics["val_dice"])
        logger.info("  Enhancing Tumor (ET) Dice: %.4f", metrics["val_dice_ET"])
        logger.info("  Tumor Core (TC) Dice:      %.4f", metrics["val_dice_TC"])
        logger.info("  Whole Tumor (WT) Dice:     %.4f", metrics["val_dice_WT"])
    else:
        logger.info("Skipping metrics evaluation (--vis_only flag set).")

    # Visualization
    if args.vis_cases > 0:
        logger.info("Generating %d visualizations...", args.vis_cases)
        vis_dir = run_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        
        # We can just iterate the val_dataset directly
        dataset = val_loader.dataset
        vis_ids = val_ids[:args.vis_cases]
        
        for cid in vis_ids:
            # Find the item index for this case_id
            item_idx = -1
            for i, (idx_cid, _) in enumerate(dataset._index):
                if idx_cid == cid:
                    item_idx = i
                    break
            
            if item_idx == -1:
                logger.warning("Case %s not found in validation index.", cid)
                continue
                
            item = dataset[item_idx]
            
            # Add batch dim
            image = item["image"].unsqueeze(0).to(device)  # (1, 3, 4, 160, 192)
            target = item["target"] # (3, 160, 192)
            slice_z = item["slice_idx"]
            
            with torch.amp.autocast(device_type="cuda"):
                model.reset_states()
                for t in range(image.shape[1]):
                    pred_t = model(image[:, t]) # (1, 3, 160, 192)
            
            pred_bin = (pred_t.detach() > 0.5).squeeze(0).cpu().numpy() # (3, H, W)
            target_bin = target.numpy()
            
            # The final slice T-1
            raw_slice = image[0, -1].cpu().numpy() # (4, H, W)
            
            save_path = vis_dir / f"{cid}_slice{slice_z:03d}.png"
            
            plot_subregion_comparison(
                image_slice=raw_slice,
                target_multi=target_bin,
                pred_multi=pred_bin,
                case_id=cid,
                slice_idx=slice_z,
                save_path=save_path
            )
            
            logger.info("Saved visualization: %s", save_path)
            
        logger.info("All visualizations saved to %s", vis_dir)


if __name__ == "__main__":
    main()
