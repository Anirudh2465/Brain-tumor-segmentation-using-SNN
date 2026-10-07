"""
train_single_slice.py — CLI: Train Spiking U-Seg-Net in single-slice axial mode.

Each training sample is ONE 2D axial slice (4, 160, 192) rather than a full
T=152 slice sequence.  FPTT is replaced by standard Adam.  After training,
overlay PNGs are saved for `--vis_cases` validation subjects.

Usage examples
--------------
Smoke test (5 subjects, 3 epochs, 2 vis subjects):
    python scripts/train_single_slice.py \\
        --max_subjects 5 --epochs 3 --vis_cases 2

Full run, fold 0:
    python scripts/train_single_slice.py --fold 0

Full run with custom data dir:
    python scripts/train_single_slice.py \\
        --data_dir /path/to/data --fold 0 --epochs 50
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

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import numpy as np

from spiking_useg.data.slice_dataset import build_single_slice_dataloader
from spiking_useg.data.dataset import load_split
from spiking_useg.models.spiking_unet import SpikingUSegNet
from spiking_useg.training.train_slice import train_slices
from spiking_useg.inference.predict_slice import predict_best_slice
from spiking_useg.utils.logging import setup_logging
from spiking_useg.utils.seed import seed_everything
from spiking_useg.utils.viz import plot_slice_overlay


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Spiking U-Seg-Net in single-slice axial mode."
    )
    # Data
    p.add_argument("--data_dir",        type=Path, default=Path("data"),
                   help="Root data directory containing processed/ and splits/.")
    p.add_argument("--experiments_dir", type=Path, default=Path("experiments"),
                   help="Root directory for experiment outputs.")
    p.add_argument("--dataset",         type=str,  default="brats23",
                   help="Dataset name, used to locate split JSON files.")
    p.add_argument("--fold",            type=int,  default=0,
                   help="CV fold index (0-based).")
    p.add_argument("--max_subjects",    type=int,  default=0,
                   help="Limit subjects for debugging (0 = use all).")
    p.add_argument("--min_tumor_frac",  type=float, default=0.001,
                   help="Min tumour pixel fraction to keep a slice (default 0.1%%).")

    # Training
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--lr",          type=float, default=0.001)
    p.add_argument("--batch_size",  type=int,   default=16)
    p.add_argument("--patience",    type=int,   default=10,
                   help="Early stopping patience (epochs).")
    p.add_argument("--grad_clip",   type=float, default=1.0,
                   help="Gradient clipping max norm (0 = disabled).")
    p.add_argument("--weight_decay",type=float, default=1e-5)
    p.add_argument("--no_augment",  action="store_true",
                   help="Disable training augmentation.")
    p.add_argument("--no_tensorboard", action="store_true")
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader workers (0 = main process; safe on Windows).")

    # Visualization
    p.add_argument("--vis_cases",   type=int,  default=5,
                   help="Number of val subjects to visualize after training (0 = none).")

    # Misc
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ── Visualization helper ──────────────────────────────────────────────────────

def run_visualization(
    model: SpikingUSegNet,
    val_ids: list[str],
    processed_dir: Path,
    vis_dir: Path,
    device: torch.device,
    n_cases: int,
) -> None:
    """For `n_cases` validation subjects, find best axial slice, predict, save PNG.

    Args:
        model: Trained SpikingUSegNet already loaded with best weights.
        val_ids: List of all validation case IDs.
        processed_dir: Root of processed data.
        vis_dir: Output directory for PNG files.
        device: Inference device.
        n_cases: How many subjects to visualize.
    """
    logger = logging.getLogger(__name__)
    vis_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    # Pick subjects that actually have a seg.npy (ground truth available)
    subjects_done = 0
    for case_id in val_ids:
        if subjects_done >= n_cases:
            break

        case_dir  = processed_dir / case_id
        data_path = case_dir / "data.npy"
        seg_path  = case_dir / "seg.npy"

        if not data_path.exists():
            logger.warning("data.npy missing for %s — skipping vis.", case_id)
            continue

        logger.info("Visualizing %s …", case_id)

        data = np.load(str(data_path))  # (4, 160, 192, 152)
        seg  = np.load(str(seg_path)) if seg_path.exists() else None  # (160, 192, 152)

        result = predict_best_slice(model, data, seg, device, threshold=0.5)

        save_path = vis_dir / f"{case_id}_slice{result['slice_idx']:03d}.png"
        plot_slice_overlay(
            image_slice   = result["image_slice"],
            gt_seg_slice  = result["gt_seg_slice"],
            pred_brats    = result["pred_brats"],
            case_id       = case_id,
            slice_idx     = result["slice_idx"],
            save_path     = save_path,
            modality_idx  = 1,  # T1c background
        )
        logger.info("  Saved overlay -> %s", save_path)
        subjects_done += 1

    logger.info("Visualization complete: %d PNGs saved to %s", subjects_done, vis_dir)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    # Output directory for this run
    run_dir = (
        args.experiments_dir
        / args.dataset
        / f"fold{args.fold}"
        / "axial_slice_attention"
    )
    setup_logging(log_dir=run_dir, run_name="train_slice", level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("  Spiking U-Seg-Net — Single-Slice Axial Mode")
    logger.info("  Dataset=%s  Fold=%d  Branch=single-slice-axial", args.dataset, args.fold)
    logger.info("=" * 60)

    # ── Load fold split ──────────────────────────────────────────────────
    splits_dir = args.data_dir / "splits"
    split = load_split(splits_dir, args.dataset, args.fold)
    train_ids: list[str] = split["train"]
    val_ids:   list[str] = split["val"]

    if args.max_subjects > 0:
        train_ids = train_ids[: args.max_subjects]
        val_ids   = val_ids[: max(1, args.max_subjects // 5)]
        logger.info(
            "DEBUG MODE: %d train / %d val subjects.",
            len(train_ids), len(val_ids),
        )

    processed_dir = args.data_dir / "processed"

    # ── Pre-flight: check processed data exists ──────────────────────────
    if not processed_dir.exists() or not any(processed_dir.iterdir()):
        logger.error(
            "\n"
            "  ╔══════════════════════════════════════════════════════════╗\n"
            "  ║  PREPROCESSED DATA NOT FOUND                            ║\n"
            "  ║                                                          ║\n"
            "  ║  Expected directory: %s\n"
            "  ║                                                          ║\n"
            "  ║  Run preprocessing first:                               ║\n"
            "  ║    python scripts/run_preprocess.py --subset 5          ║\n"
            "  ║  (--subset 5 processes only 5 cases for a quick test)   ║\n"
            "  ╚══════════════════════════════════════════════════════════╝",
            processed_dir,
        )
        sys.exit(1)

    # ── Build DataLoaders ────────────────────────────────────────────────
    train_loader = build_single_slice_dataloader(
        processed_dir     = processed_dir,
        case_ids          = train_ids,
        batch_size        = args.batch_size,
        shuffle           = True,
        augment           = not args.no_augment,
        min_tumor_fraction= args.min_tumor_frac,
        require_seg       = True,
        num_workers       = args.num_workers,
    )
    val_loader = build_single_slice_dataloader(
        processed_dir     = processed_dir,
        case_ids          = val_ids,
        batch_size        = max(1, args.batch_size // 2),
        shuffle           = False,
        augment           = False,
        min_tumor_fraction= args.min_tumor_frac,
        require_seg       = True,
        num_workers       = args.num_workers,
    )

    # ── Check datasets are non-empty ─────────────────────────────────────
    if len(train_loader.dataset) == 0:
        logger.error(
            "Training dataset is empty! All %d subject(s) were skipped.\n"
            "  -> Check that preprocessing ran for these case IDs:\n"
            "      %s ...\n"
            "  -> Run: python scripts/run_preprocess.py --subset %d",
            len(train_ids), train_ids[:3], len(train_ids),
        )
        sys.exit(1)
    if len(val_loader.dataset) == 0:
        logger.error(
            "Validation dataset is empty! Check that preprocessing ran for val subjects.\n"
            "  → Run: python scripts/run_preprocess.py --subset %d",
            len(train_ids) + len(val_ids),
        )
        sys.exit(1)

    logger.info(
        "DataLoaders ready: %d train slices / %d val slices.",
        len(train_loader.dataset),
        len(val_loader.dataset),
    )

    # ── Build model ──────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = SpikingUSegNet()

    if torch.cuda.device_count() > 1:
        logger.info("Using %d GPUs with DataParallel.", torch.cuda.device_count())
        model = torch.nn.DataParallel(model)

    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %d (%.2f M)", n_params, n_params / 1e6)

    # ── Train ────────────────────────────────────────────────────────────
    results = train_slices(
        model           = model,
        train_loader    = train_loader,
        val_loader      = val_loader,
        run_dir         = run_dir,
        max_epochs      = args.epochs,
        lr              = args.lr,
        weight_decay    = args.weight_decay,
        grad_clip_norm  = args.grad_clip,
        patience        = args.patience,
        device          = device,
        use_tensorboard = not args.no_tensorboard,
    )

    logger.info("Training done. Best val_dice = %.4f", results["best_val_dice"])

    # ── Visualization ────────────────────────────────────────────────────
    if args.vis_cases > 0:
        logger.info("Loading best model for visualization …")
        best_path = run_dir / "best_model.pt"
        if best_path.exists():
            ckpt = torch.load(str(best_path), map_location=device)
            # Handle DataParallel wrapper
            if hasattr(model, "module"):
                model.module.load_state_dict(ckpt["model_state_dict"])
            else:
                model.load_state_dict(ckpt["model_state_dict"])
        else:
            logger.warning("best_model.pt not found — using current model weights.")

        vis_dir = run_dir / "visualizations"
        run_visualization(
            model         = model.module if hasattr(model, "module") else model,
            val_ids       = val_ids,
            processed_dir = processed_dir,
            vis_dir       = vis_dir,
            device        = device,
            n_cases       = args.vis_cases,
        )


if __name__ == "__main__":
    main()
