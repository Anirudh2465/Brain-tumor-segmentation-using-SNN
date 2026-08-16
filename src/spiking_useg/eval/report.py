"""
report.py — Generate evaluation tables from experiments/ metrics JSON files.

Reproduces Tables 1–3 from arXiv:2601.16652v1:
  Table 1: Dice scores (ET / TC / WT) per view + ensemble, mean ± std over 5 folds.
  Table 2: NLL per view + ensemble.
  Table 3: FLOPs breakdown (loaded from separate flops JSON).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_fold_metrics(experiments_dir: Path | str, dataset: str) -> list[dict]:
    """Load per-fold metrics from JSON files.

    Expects structure:
        experiments_dir/
            {dataset}/
                fold{k}/
                    {view}/
                        metrics.json

    Args:
        experiments_dir: Root experiments directory.
        dataset: Dataset name (e.g., 'brats23').

    Returns:
        List of metric dicts, each with keys: fold, view, best_val_dice, etc.
    """
    experiments_dir = Path(experiments_dir)
    records = []
    dataset_dir = experiments_dir / dataset
    if not dataset_dir.exists():
        return records

    for fold_dir in sorted(dataset_dir.iterdir()):
        if not fold_dir.is_dir() or not fold_dir.name.startswith("fold"):
            continue
        fold_idx = int(fold_dir.name.replace("fold", ""))
        for view_dir in sorted(fold_dir.iterdir()):
            if not view_dir.is_dir():
                continue
            view = view_dir.name
            metrics_path = view_dir / "metrics.json"
            if not metrics_path.exists():
                continue
            with open(metrics_path) as f:
                metrics = json.load(f)
            records.append({
                "fold": fold_idx,
                "view": view,
                **{k: v for k, v in metrics.items() if k != "history"},
            })

    return records


def load_ensemble_metrics(experiments_dir: Path | str, dataset: str) -> list[dict]:
    """Load per-fold ensemble metrics JSON files.

    Expects:
        experiments_dir/{dataset}/fold{k}/ensemble/metrics.json
    """
    experiments_dir = Path(experiments_dir)
    records = []
    for fold_dir in sorted((experiments_dir / dataset).glob("fold*")):
        fold_idx = int(fold_dir.name.replace("fold", ""))
        ensemble_path = fold_dir / "ensemble" / "metrics.json"
        if ensemble_path.exists():
            with open(ensemble_path) as f:
                metrics = json.load(f)
            records.append({"fold": fold_idx, "view": "ensemble", **metrics})
    return records


def generate_dice_table(experiments_dir: str | Path, dataset: str = "brats23") -> pd.DataFrame:
    """Generate Table 1: Dice mean ± std per view and ensemble.

    Returns:
        DataFrame with columns: view, ET, TC, WT, mean_dice
    """
    records = load_fold_metrics(experiments_dir, dataset)
    records += load_ensemble_metrics(experiments_dir, dataset)

    if not records:
        print("No metrics found. Run training and ensemble evaluation first.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    # Summarize per view
    rows = []
    for view in sorted(df["view"].unique()):
        vdf = df[df["view"] == view]
        for col in ["val_dice_ET", "val_dice_TC", "val_dice_WT", "best_val_dice"]:
            if col in vdf.columns:
                mean = vdf[col].mean()
                std = vdf[col].std()
                rows.append({"view": view, "metric": col, "mean": mean, "std": std})

    summary = pd.DataFrame(rows)
    print("\n=== Table 1: Dice Scores ===")
    print(summary.to_string(index=False))
    return summary


def generate_nll_table(experiments_dir: str | Path, dataset: str = "brats23") -> pd.DataFrame:
    """Generate Table 2: NLL per view and ensemble."""
    records = load_fold_metrics(experiments_dir, dataset)
    records += load_ensemble_metrics(experiments_dir, dataset)
    df = pd.DataFrame(records)

    nll_cols = [c for c in df.columns if "nll" in c.lower()]
    if not nll_cols:
        print("No NLL metrics found.")
        return pd.DataFrame()

    rows = []
    for view in sorted(df["view"].unique()):
        vdf = df[df["view"] == view]
        for col in nll_cols:
            rows.append({
                "view": view,
                "metric": col,
                "mean": vdf[col].mean(),
                "std": vdf[col].std(),
            })

    summary = pd.DataFrame(rows)
    print("\n=== Table 2: NLL Scores ===")
    print(summary.to_string(index=False))
    return summary


def generate_flops_table(experiments_dir: str | Path, dataset: str = "brats23") -> pd.DataFrame:
    """Generate Table 3: FLOPs per view."""
    experiments_dir = Path(experiments_dir)
    records = []
    dataset_dir = experiments_dir / dataset
    if not dataset_dir.exists():
        return pd.DataFrame()

    for fold_dir in sorted(dataset_dir.iterdir()):
        if not fold_dir.is_dir() or not fold_dir.name.startswith("fold"):
            continue
        for view_dir in sorted(fold_dir.iterdir()):
            if not view_dir.is_dir():
                continue
            view = view_dir.name
            flops_path = view_dir / "flops.json"
            if not flops_path.exists():
                continue
            with open(flops_path) as f:
                flops = json.load(f)
            records.append({"view": view, **flops})

    if not records:
        print("\nNo FLOPs metrics found. Ensure train_single_view saves flops.json.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    rows = []
    for view in sorted(df["view"].unique()):
        vdf = df[df["view"] == view]
        for col in ["dense_flops_per_step", "mean_spike_rate", "effective_flops_per_step", "effective_flops_per_volume"]:
            if col in vdf.columns:
                rows.append({
                    "view": view,
                    "metric": col,
                    "mean": vdf[col].mean(),
                    "std": vdf[col].std(),
                })

    summary = pd.DataFrame(rows)
    print("\n=== Table 3: FLOPs Scores ===")
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    import sys
    exp_dir = sys.argv[1] if len(sys.argv) > 1 else "experiments"
    dataset = sys.argv[2] if len(sys.argv) > 2 else "brats23"
    generate_dice_table(exp_dir, dataset)
    generate_nll_table(exp_dir, dataset)
    generate_flops_table(exp_dir, dataset)
