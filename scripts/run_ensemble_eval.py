"""
run_ensemble_eval.py — CLI: Run multi-view ensemble and compute Dice + NLL.

Loads one trained model per view for a given fold, runs full ensemble inference
on all validation subjects, and saves metrics to experiments/{dataset}/fold{k}/ensemble/metrics.json.

Usage:
    python scripts/run_ensemble_eval.py --fold 0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch

from spiking_useg.data.dataset import load_split
from spiking_useg.eval.metrics import dice_score_numpy, nll_score, aggregate_fold_metrics
from spiking_useg.inference.ensemble import ensemble_predict
from spiking_useg.utils.logging import setup_logging
from spiking_useg.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-view ensemble evaluation")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--data_dir", type=Path, default=Path("data"))
    p.add_argument("--experiments_dir", type=Path, default=Path("experiments"))
    p.add_argument("--dataset", type=str, default="brats23")
    p.add_argument("--views", nargs="+", default=["axial", "coronal", "sagittal"])
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    out_dir = args.experiments_dir / args.dataset / f"fold{args.fold}" / "ensemble"
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir=out_dir, run_name="ensemble_eval")
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed_dir = args.data_dir / "processed"

    # Load split
    split = load_split(args.data_dir / "splits", args.dataset, args.fold)
    val_ids = split["val"]
    logger.info("Evaluating %d validation subjects for fold %d ...", len(val_ids), args.fold)

    # Locate checkpoints
    view_checkpoints = {}
    for view in args.views:
        ckpt = args.experiments_dir / args.dataset / f"fold{args.fold}" / view / "best_model.pt"
        if not ckpt.exists():
            logger.error("Checkpoint not found: %s — skip view %s", ckpt, view)
            continue
        view_checkpoints[view] = ckpt

    if not view_checkpoints:
        logger.error("No view checkpoints found. Train models first.")
        sys.exit(1)

    logger.info("Using views: %s", list(view_checkpoints.keys()))

    # ── Evaluate per subject ──────────────────────────────────────────────
    per_subject_metrics = []

    for case_id in val_ids:
        case_dir = processed_dir / case_id
        if not (case_dir / "data.npy").exists():
            logger.warning("Missing data for %s — skipping.", case_id)
            continue
        if not (case_dir / "seg.npy").exists():
            logger.warning("Missing seg for %s — skipping.", case_id)
            continue

        data = np.load(str(case_dir / "data.npy"))   # (4, H, W, D)
        seg = np.load(str(case_dir / "seg.npy"))     # (H, W, D)

        # Derive binary GT from raw seg
        gt_et = (seg == 3).astype(np.float32)
        gt_tc = ((seg == 1) | (seg == 3)).astype(np.float32)
        gt_wt = (seg > 0).astype(np.float32)
        gt = np.stack([gt_et, gt_tc, gt_wt], axis=0)  # (3, H, W, D)

        # Run ensemble
        result = ensemble_predict(
            view_checkpoints=view_checkpoints,
            data=data,
            device=device,
            threshold=args.threshold,
        )

        pred_probs = np.stack([result["prob_ET"], result["prob_TC"], result["prob_WT"]], axis=0)
        pred_bin = (pred_probs > args.threshold).astype(np.float32)

        dice = dice_score_numpy(pred_bin, gt)
        nll = nll_score(pred_probs, gt)

        subject_metrics = {
            "case_id": case_id,
            "dice_ET": float(dice[0]),
            "dice_TC": float(dice[1]),
            "dice_WT": float(dice[2]),
            "nll_ET": float(nll[0]),
            "nll_TC": float(nll[1]),
            "nll_WT": float(nll[2]),
        }
        per_subject_metrics.append(subject_metrics)
        logger.info(
            "  %s  Dice: ET=%.3f TC=%.3f WT=%.3f  NLL: ET=%.3f TC=%.3f WT=%.3f",
            case_id,
            dice[0], dice[1], dice[2],
            nll[0], nll[1], nll[2],
        )

    # Aggregate
    numeric_metrics = [
        {k: v for k, v in m.items() if k != "case_id"}
        for m in per_subject_metrics
    ]
    agg = aggregate_fold_metrics(numeric_metrics)

    logger.info("-" * 60)
    logger.info("Ensemble Fold %d Summary:", args.fold)
    for k, v in agg.items():
        logger.info("  %s: %.4f", k, v)

    # Save
    out = {
        "fold": args.fold,
        "n_subjects": len(per_subject_metrics),
        "views_used": list(view_checkpoints.keys()),
        "aggregated": agg,
        "per_subject": per_subject_metrics,
    }
    out_path = out_dir / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Saved ensemble metrics -> %s", out_path)


if __name__ == "__main__":
    main()
