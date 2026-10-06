"""
train_2_5d.py — Training loop for 2.5D Causal mode (T=3 slices).
"""

from __future__ import annotations

import logging

import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from spiking_useg.eval.metrics import dice_score
from spiking_useg.training.losses import hybrid_loss
from spiking_useg.training.scheduler import build_scheduler

logger = logging.getLogger(__name__)


def train_one_epoch_2_5d(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float = 1.0,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    n_batches = 0

    for batch in loader:
        # image is (B, T, 4, H, W)
        image = batch["image"].to(device)
        target = batch["target"].to(device)  # (B, 3, H, W) for the final slice T-1

        B, T, C, H, W = image.shape

        # PLIF state reset for the sequence
        if hasattr(model, "module"):
            model.module.reset_states()
        else:
            model.reset_states()

        optimizer.zero_grad()

        # Iterate through the sequence to build up temporal memory
        for t in range(T):
            x_t = image[:, t]
            pred_t = model(x_t)

        # We only care about the prediction at the final step T-1 for loss
        loss = hybrid_loss(pred_t, target)
        loss.backward()

        if grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

        optimizer.step()

        with torch.no_grad():
            pred_bin = (pred_t.detach() > 0.5).float()
            d = dice_score(pred_bin, target).mean().item()

        total_loss += loss.item()
        total_dice += d
        n_batches += 1

        if n_batches % 20 == 0 or n_batches == len(loader):
            logger.info(
                "  Train Batch %d/%d — Loss: %.4f, Dice: %.4f",
                n_batches, len(loader),
                total_loss / n_batches,
                total_dice / n_batches,
            )

    return {
        "loss": total_loss / max(n_batches, 1),
        "train_dice": total_dice / max(n_batches, 1),
    }


@torch.no_grad()
def validate_one_epoch_2_5d(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_dice_et = 0.0
    total_dice_tc = 0.0
    total_dice_wt = 0.0
    n_batches = 0

    for batch in loader:
        image = batch["image"].to(device)    # (B, T, 4, H, W)
        target = batch["target"].to(device)  # (B, 3, H, W)

        B, T, C, H, W = image.shape

        if hasattr(model, "module"):
            model.module.reset_states()
        else:
            model.reset_states()

        for t in range(T):
            x_t = image[:, t]
            pred_t = model(x_t)

        pred_bin = (pred_t > 0.5).float()
        
        # Calculate dice per class
        dice = dice_score(pred_bin, target)  # (3,)

        total_dice_et += dice[0].item()
        total_dice_tc += dice[1].item()
        total_dice_wt += dice[2].item()
        n_batches += 1

        if n_batches % 20 == 0 or n_batches == len(loader):
            logger.info("  Val Batch %d/%d processed.", n_batches, len(loader))

    return {
        "val_dice": (total_dice_et + total_dice_tc + total_dice_wt) / (3 * max(n_batches, 1)),
        "val_dice_ET": total_dice_et / max(n_batches, 1),
        "val_dice_TC": total_dice_tc / max(n_batches, 1),
        "val_dice_WT": total_dice_wt / max(n_batches, 1),
    }

def train_2_5d_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    device: torch.device,
    epochs: int,
    run_dir: Path,
    grad_clip_norm: float = 1.0,
    use_tensorboard: bool = True,
) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=str(run_dir / "tb_logs")) if use_tensorboard else None

    best_val_dice = -1.0
    metrics_history = []

    for epoch in range(1, epochs + 1):
        logger.info("-" * 40)
        logger.info("Epoch %d/%d", epoch, epochs)
        t0 = time.time()

        train_metrics = train_one_epoch_2_5d(
            model, train_loader, optimizer, device, grad_clip_norm
        )
        val_metrics = validate_one_epoch_2_5d(model, val_loader, device)

        if scheduler is not None:
            scheduler.step(val_metrics["val_dice"])

        t_end = time.time()

        # Log to tensorboard
        if tb_writer:
            tb_writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
            tb_writer.add_scalar("Dice/train", train_metrics["train_dice"], epoch)
            tb_writer.add_scalar("Dice/val_overall", val_metrics["val_dice"], epoch)
            tb_writer.add_scalar("Dice/val_ET", val_metrics["val_dice_ET"], epoch)
            tb_writer.add_scalar("Dice/val_TC", val_metrics["val_dice_TC"], epoch)
            tb_writer.add_scalar("Dice/val_WT", val_metrics["val_dice_WT"], epoch)
            tb_writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        # Print summary
        logger.info(
            "Epoch %3d | loss=%.4f | train_dice=%.4f | val_dice=%.4f | "
            "ET=%.4f | TC=%.4f | WT=%.4f | %.1fs",
            epoch,
            train_metrics["loss"],
            train_metrics["train_dice"],
            val_metrics["val_dice"],
            val_metrics["val_dice_ET"],
            val_metrics["val_dice_TC"],
            val_metrics["val_dice_WT"],
            t_end - t0,
        )

        # Save latest
        latest_path = run_dir / "latest_model.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": val_metrics["val_dice"],
            },
            str(latest_path),
        )

        # Save best
        if val_metrics["val_dice"] > best_val_dice:
            best_val_dice = val_metrics["val_dice"]
            best_path = run_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_dice": best_val_dice,
                },
                str(best_path),
            )
            logger.info("  [BEST] val_dice=%.4f — saved %s", best_val_dice, best_path)

        metrics_history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_dice": train_metrics["train_dice"],
                **val_metrics,
                "time_sec": t_end - t0,
            }
        )

    if tb_writer:
        tb_writer.close()

    return {"best_val_dice": best_val_dice, "history": metrics_history}
