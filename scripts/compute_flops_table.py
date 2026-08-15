"""
compute_flops_table.py — CLI: Compute spike-rate-aware FLOPs table (Table 3).

Usage:
    python scripts/compute_flops_table.py [options]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from spiking_useg.efficiency.flops import measure_snn_flops
from spiking_useg.models.spiking_unet import SpikingUSegNet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute spike-rate-aware FLOPs for Table 3")
    p.add_argument("--views", nargs="+", default=["axial", "coronal", "sagittal"],
                   help="Views to benchmark")
    return p.parse_args()


# View → (C, H, W) input shape
VIEW_SHAPES = {
    "axial":    (4, 160, 192),
    "coronal":  (4, 160, 152),
    "sagittal": (4, 192, 152),
}

VIEW_TIMESTEPS = {
    "axial":    152,
    "coronal":  192,
    "sagittal": 160,
}


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n=== Table 3: Spike-Rate-Aware FLOPs ===")
    print(f"{'View':<12} {'Dense FLOPs/step':>18} {'Spike Rate':>12} {'Eff FLOPs/step':>16} {'Eff FLOPs/vol':>16}")
    print("-" * 76)

    total_dense = 0
    total_eff = 0

    for view in args.views:
        if view not in VIEW_SHAPES:
            print(f"Unknown view: {view}")
            continue

        model = SpikingUSegNet()
        result = measure_snn_flops(
            model=model,
            input_shape=VIEW_SHAPES[view],
            num_timesteps=VIEW_TIMESTEPS[view],
            device=device,
        )
        T = VIEW_TIMESTEPS[view]

        dense = result["dense_flops_per_step"]
        rate = result["mean_spike_rate"]
        eff_step = result["effective_flops_per_step"]
        eff_vol = result["effective_flops_per_volume"]

        print(
            f"{view:<12} {dense:>18,} {rate:>12.3f} {eff_step:>16,.0f} {eff_vol:>16,.0f}"
        )
        if dense > 0:
            total_dense += dense * T
            total_eff += eff_vol if eff_vol > 0 else 0

    print("-" * 76)
    if total_dense > 0:
        reduction = 1.0 - total_eff / total_dense
        print(f"{'Ensemble total':<12} {total_dense:>18,} {'':>12} {total_eff:>16,.0f}")
        print(f"\nFLOPs reduction vs. dense SNN: {reduction:.1%} (paper reports ~87%)")


if __name__ == "__main__":
    main()
