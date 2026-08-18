"""
train_loop.py — Per-view FPTT training loop with 5-fold CV driver.

Flow per epoch:
  for each subject-sequence in train DataLoader:
      model.reset_states()
      fptt.start_sequence()
      for t, (x_t, y_t) in enumerate(slices):
          pred_t = model(x_t)
          loss_t = hybrid_loss(pred_t, y_t)
          fptt.step(loss_t)   # handles backward + FPTT + Adam

Validation:
  Run full subject sequences through model (no grad), accumulate Dice per slice,
  average across slices then subjects → val_dice.

Early stopping: stop if val_dice doesn't improve for `patience` epochs.
Save checkpoint when val_dice improves.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from spiking_useg.eval.metrics import dice_score
from spiking_useg.training.fptt import FPTTOptimizer
from spiking_useg.training.losses import hybrid_loss
from spiking_useg.training.scheduler import build_scheduler

logger = logging.getLogger(__name__)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    fptt: FPTTOptimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    n_batches = 0

    for batch in loader:
        images = batch["images"].to(device)   # (B, T, 4, H, W)
        targets = batch["targets"].to(device)  # (B, T, 3, H, W)

        B, T, C, H, W = images.shape

        if hasattr(model, 'module'):
            model.module.reset_states()
        else:
            model.reset_states()
            
        fptt.start_sequence()

        slice_dice_sum = 0.0
        for t in range(T):
            x_t = images[:, t]   # (B, 4, H, W)
            y_t = targets[:, t]  # (B, 3, H, W)

            pred_t = model(x_t)              # (B, 3, H, W)
            loss_t = hybrid_loss(pred_t, y_t)

            step_loss = fptt.step(loss_t)
            total_loss += step_loss

            with torch.no_grad():
                d = dice_score(pred_t.detach(), y_t).mean().item()
                slice_dice_sum += d

        total_dice += slice_dice_sum / T
        n_batches += 1

        if n_batches % 5 == 0 or n_batches == len(loader):
            cur_loss = total_loss / (n_batches * T)
            cur_dice = total_dice / n_batches
            logger.info("  Train Batch %d/%d - Loss: %.4f, Dice: %.4f", n_batches, len(loader), cur_loss, cur_dice)

    mean_loss = total_loss / max(n_batches * T, 1)
    mean_dice = total_dice / max(n_batches, 1)
    return {"loss": mean_loss, "train_dice": mean_dice}

@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    dice_et = dice_tc = dice_wt = 0.0
    n_batches = 0

    for batch in loader:
        images = batch["images"].to(device)
        targets = batch["targets"].to(device)
        B, T, C, H, W = images.shape

        if hasattr(model, 'module'):
            model.module.reset_states()
        else:
            model.reset_states()

        all_preds = []
        for t in range(T):
            x_t = images[:, t]
            pred_t = model(x_t)  # (B, 3, H, W)
            all_preds.append(pred_t.unsqueeze(1)) # (B, 1, 3, H, W)

        preds = torch.cat(all_preds, dim=1)   # (B, T, 3, H, W)
        preds_bin = (preds > 0.5).float()

        d = dice_score(preds_bin.view(B * T, 3, H, W), targets.view(B * T, 3, H, W))
        dice_et += d[0].item()
        dice_tc += d[1].item()
        dice_wt += d[2].item()
        n_batches += 1
        
        if n_batches % 5 == 0 or n_batches == len(loader):
            logger.info("  Val Batch %d/%d processed", n_batches, len(loader))

    n = max(n_batches, 1)
    return {
        "val_dice_ET": dice_et / n,
        "val_dice_TC": dice_tc / n,
        "val_dice_WT": dice_wt / n,
        "val_dice": (dice_et + dice_tc + dice_wt) / (3 * n),
    }


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    run_dir: Path,
    max_epochs: int = 100,
    lr: float = 0.001,
    weight_decay: float = 1e-5,
    fptt_alpha: float = 0.1,
    grad_clip_norm: float = 0.3,
    patience: int = 10,
    device: Optional[torch.device] = None,
    use_tensorboard: bool = True,
) -> dict:
    """Full training run for one (view x fold) combination.

    Args:
        model: SpikingUSegNet model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        run_dir: Directory to save checkpoints and metrics.
        max_epochs: Maximum number of epochs.
        lr: Adam learning rate.
        weight_decay: Adam weight decay.
        fptt_alpha: FPTT regularization strength.
        grad_clip_norm: Gradient clipping max norm.
        patience: Early stopping patience in epochs.
        device: Torch device (auto-detected if None).
        use_tensorboard: Whether to log to TensorBoard.

    Returns:
        Dict with best validation metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Training run directory: %s", run_dir)
    logger.info("Device: %s", device)

    model = model.to(device)

    base_optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    fptt = FPTTOptimizer(base_optim, alpha=fptt_alpha, grad_clip_norm=grad_clip_norm)
    scheduler = build_scheduler(base_optim)

    writer: Optional[SummaryWriter] = None
    if use_tensorboard:
        writer = SummaryWriter(log_dir=str(run_dir / "tb_logs"))

    best_val_dice = -1.0
    no_improve_count = 0
    start_epoch = 1
    history: list[dict] = []

    # Check for existing checkpoint to resume
    latest_ckpt_path = run_dir / "latest_model.pt"
    if latest_ckpt_path.exists():
        logger.info("Found existing checkpoint: %s. Resuming training...", latest_ckpt_path)
        checkpoint = torch.load(latest_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        base_optim.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_dice = checkpoint.get("best_val_dice", -1.0)
        no_improve_count = checkpoint.get("no_improve_count", 0)
        logger.info("Resuming from epoch %d with best_val_dice %.4f", start_epoch, best_val_dice)

    for epoch in range(start_epoch, max_epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(model, train_loader, fptt, device)
        val_metrics = validate_one_epoch(model, val_loader, device)

        elapsed = time.time() - t0
        val_dice = val_metrics["val_dice"]

        # LR scheduling on val Dice
        scheduler.step(val_dice)

        epoch_record = {
            "epoch": epoch,
            "elapsed_s": elapsed,
            **train_metrics,
            **val_metrics,
        }
        history.append(epoch_record)

        logger.info(
            "Epoch %3d | loss=%.4f | train_dice=%.4f | val_dice=%.4f"
            " | ET=%.4f | TC=%.4f | WT=%.4f | %.1fs",
            epoch,
            train_metrics["loss"],
            train_metrics["train_dice"],
            val_dice,
            val_metrics["val_dice_ET"],
            val_metrics["val_dice_TC"],
            val_metrics["val_dice_WT"],
            elapsed,
        )

        if writer:
            for k, v in epoch_record.items():
                if isinstance(v, float):
                    writer.add_scalar(k, v, epoch)

        # Save best checkpoint
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            no_improve_count = 0
            ckpt_path = run_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_metrics": val_metrics,
                },
                ckpt_path,
            )
            logger.info("  [BEST] New best val_dice=%.4f -- checkpoint saved.", best_val_dice)
        else:
            no_improve_count += 1
            logger.info("  No improvement for %d epochs.", no_improve_count)

        # Save latest checkpoint for resuming
        latest_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": base_optim.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_dice": best_val_dice,
            "no_improve_count": no_improve_count,
            "val_metrics": val_metrics,
        }
        torch.save(latest_state, latest_ckpt_path)

        # Early stopping
        if no_improve_count >= patience:
            logger.info(
                "Early stopping at epoch %d (no improvement for %d epochs).",
                epoch, patience,
            )
            break

    # Save metrics JSON
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"history": history, "best_val_dice": best_val_dice}, f, indent=2)

    if writer:
        writer.close()

    logger.info("Training complete. Best val_dice=%.4f", best_val_dice)
    return {"best_val_dice": best_val_dice, "history": history}
