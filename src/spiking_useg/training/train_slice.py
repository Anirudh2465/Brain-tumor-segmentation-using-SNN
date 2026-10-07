"""
train_slice.py — Simplified training loop for single-slice axial mode.

Key differences from train_loop.py (sequence FPTT mode):
  - No FPTT: standard Adam + hybrid_loss with regular .backward()
  - model.reset_states() called once per batch (not per subject sequence)
  - DataLoader yields flat (B, 4, H, W) batches, not (B, T, 4, H, W) sequences
  - Validation Dice computed directly on thresholded predictions

Flow per epoch:
    for batch in train_loader:
        image  = batch["image"].to(device)    # (B, 4, H, W)
        target = batch["target"].to(device)   # (B, 3, H, W)
        model.reset_states()
        pred = model(image)                   # (B, 3, H, W)
        loss = hybrid_loss(pred, target)
        loss.backward()
        optimizer.step()
"""

from __future__ import annotations

import json
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


# ── Per-epoch helpers ─────────────────────────────────────────────────────────

def train_one_epoch_slices(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
    grad_clip_norm: float = 1.0,
) -> dict[str, float]:
    """Train for one epoch over individual slices.

    Args:
        model: SpikingUSegNet.
        loader: DataLoader yielding batches with 'image' (B,4,H,W) and 'target' (B,3,H,W).
        optimizer: Adam optimizer.
        device: Torch device.
        grad_clip_norm: Max gradient norm for clipping (0 = disabled).

    Returns:
        Dict with 'loss' and 'train_dice' (mean over all classes).
    """
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    n_batches = 0

    for batch in loader:
        image  = batch["image"].to(device)    # (B, 4, H, W)
        target = batch["target"].to(device)   # (B, 3, H, W)

        # PLIF state reset: treat each slice independently (T=1 SNN step)
        if hasattr(model, "module"):
            model.module.reset_states()
        else:
            model.reset_states()

        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.autocast(device_type="cuda"):
                pred = model(image)
            # Compute loss in float32 for numeric stability (AMP safety)
            loss = hybrid_loss(pred.float(), target.float())
            scaler.scale(loss).backward()
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(image)
            loss = hybrid_loss(pred, target)
            loss.backward()
            if grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

        with torch.no_grad():
            pred_bin = (pred.detach() > 0.5).float()
            d = dice_score(pred_bin, target).mean().item()

        total_loss += loss.item()
        total_dice += d
        n_batches  += 1

        if n_batches % 20 == 0 or n_batches == len(loader):
            logger.info(
                "  Train Batch %d/%d — Loss: %.4f, Dice: %.4f",
                n_batches, len(loader),
                total_loss / n_batches,
                total_dice / n_batches,
            )

    return {
        "loss":       total_loss / max(n_batches, 1),
        "train_dice": total_dice / max(n_batches, 1),
    }


@torch.no_grad()
def validate_one_epoch_slices(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Validate for one epoch over individual slices.

    Returns:
        Dict with per-class val Dice (val_dice_ET, val_dice_TC, val_dice_WT)
        and their mean (val_dice).
    """
    model.eval()
    dice_et = dice_tc = dice_wt = 0.0
    n_batches = 0

    for batch in loader:
        image  = batch["image"].to(device)
        target = batch["target"].to(device)

        if hasattr(model, "module"):
            model.module.reset_states()
        else:
            model.reset_states()

        pred = model(image)                    # (B, 3, H, W) — probabilities
        pred_bin = (pred > 0.5).float()

        d = dice_score(pred_bin, target)       # (3,)
        dice_et += d[0].item()
        dice_tc += d[1].item()
        dice_wt += d[2].item()
        n_batches += 1

        if n_batches % 20 == 0 or n_batches == len(loader):
            logger.info("  Val Batch %d/%d processed.", n_batches, len(loader))

    n = max(n_batches, 1)
    return {
        "val_dice_ET": dice_et / n,
        "val_dice_TC": dice_tc / n,
        "val_dice_WT": dice_wt / n,
        "val_dice":    (dice_et + dice_tc + dice_wt) / (3 * n),
    }


# ── Main training driver ──────────────────────────────────────────────────────

def train_slices(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    run_dir: Path,
    max_epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 1e-5,
    grad_clip_norm: float = 1.0,
    patience: int = 10,
    device: Optional[torch.device] = None,
    use_tensorboard: bool = True,
) -> dict:
    """Full training run for single-slice axial mode.

    Args:
        model: SpikingUSegNet.
        train_loader: Training DataLoader (single slices).
        val_loader: Validation DataLoader (single slices).
        run_dir: Directory for checkpoints, logs, and metrics.
        max_epochs: Maximum number of epochs.
        lr: Adam learning rate.
        weight_decay: Adam weight decay.
        grad_clip_norm: Gradient clipping norm (0 = disabled).
        patience: Early stopping patience in epochs.
        device: Torch device (auto-detected if None).
        use_tensorboard: Log scalars to TensorBoard.

    Returns:
        Dict with 'best_val_dice' and 'history'.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run directory : %s", run_dir)
    logger.info("Device        : %s", device)

    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = build_scheduler(optimizer)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    writer: Optional[SummaryWriter] = None
    if use_tensorboard:
        writer = SummaryWriter(log_dir=str(run_dir / "tb_logs"))

    best_val_dice  = -1.0
    no_improve     = 0
    start_epoch    = 1
    history: list[dict] = []

    # ── Resume from latest checkpoint if it exists ────────────────────────
    latest_path = run_dir / "latest_model.pt"
    if latest_path.exists():
        logger.info("Resuming from %s …", latest_path)
        ckpt = torch.load(latest_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_dice = ckpt.get("best_val_dice", -1.0)
        no_improve    = ckpt.get("no_improve_count", 0)
        logger.info(
            "Resuming from epoch %d | best_val_dice=%.4f", start_epoch, best_val_dice
        )

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(start_epoch, max_epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch_slices(
            model, train_loader, optimizer, device, scaler, grad_clip_norm
        )
        val_metrics = validate_one_epoch_slices(model, val_loader, device)

        elapsed    = time.time() - t0
        val_dice   = val_metrics["val_dice"]
        scheduler.step(val_dice)

        record = {"epoch": epoch, "elapsed_s": elapsed, **train_metrics, **val_metrics}
        history.append(record)

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
            for k, v in record.items():
                if isinstance(v, float):
                    writer.add_scalar(k, v, epoch)

        # ── Save best checkpoint ──────────────────────────────────────────
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            no_improve    = 0
            best_path = run_dir / "best_model.pt"
            torch.save(
                {
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "val_metrics":      val_metrics,
                },
                best_path,
            )
            logger.info("  [BEST] val_dice=%.4f — saved %s", best_val_dice, best_path)
        else:
            no_improve += 1
            logger.info("  No improvement for %d epoch(s).", no_improve)

        # ── Save latest checkpoint (for resume) ───────────────────────────
        torch.save(
            {
                "epoch":               epoch,
                "model_state_dict":    model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_dice":       best_val_dice,
                "no_improve_count":    no_improve,
                "val_metrics":         val_metrics,
            },
            latest_path,
        )

        # ── Early stopping ────────────────────────────────────────────────
        if no_improve >= patience:
            logger.info(
                "Early stopping at epoch %d (no improvement for %d epochs).",
                epoch, patience,
            )
            break

    # ── Save metrics JSON ─────────────────────────────────────────────────
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"history": history, "best_val_dice": best_val_dice}, f, indent=2)

    if writer:
        writer.close()

    logger.info("Training complete. Best val_dice=%.4f", best_val_dice)
    return {"best_val_dice": best_val_dice, "history": history}
