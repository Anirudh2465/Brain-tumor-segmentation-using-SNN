# compare_models.py
from __future__ import annotations
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from spiking_useg.models.spiking_unet import SpikingUSegNet
from spiking_useg.data.dataset_2_5d import build_2_5d_dataloader
from spiking_useg.data.dataset import load_split

def compute_dice(pred, target):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    if union == 0:
        return 1.0
    return (2. * intersection / union)

_OVERLAY_COLORS = {
    1: (1.0, 0.2, 0.2, 0.75),   # TC core — red
    2: (1.0, 1.0, 0.2, 0.65),   # WT rim  — yellow
    3: (0.2, 1.0, 0.2, 0.85),   # ET      — green
}

def to_brats_map(pred_multi):
    """Convert (3, H, W) binary predictions to a single (H, W) label map (1=TC, 2=WT, 3=ET)"""
    et, tc, wt = pred_multi[0], pred_multi[1], pred_multi[2]
    # Reconstruct mutually exclusive labels for visualization
    res = np.zeros_like(et, dtype=np.int32)
    res[wt == 1] = 2  # WT base
    res[tc == 1] = 1  # TC overwrites WT
    res[et == 1] = 3  # ET overwrites TC
    return res

def create_rgba_overlay(label_map):
    H, W = label_map.shape
    overlay = np.zeros((H, W, 4), dtype=np.float32)
    for label, color in _OVERLAY_COLORS.items():
        mask = label_map == label
        for c in range(4):
            overlay[mask, c] = color[c]
    return overlay

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path("data")
    split = load_split(data_dir / "splits", "brats23", 0)
    val_ids = split["val"][:3]
    
    loader = build_2_5d_dataloader(
        processed_dir=data_dir / "processed",
        case_ids=val_ids,
        batch_size=1,
        shuffle=False,
        min_tumor_fraction=0.01,
        require_seg=True,
        num_workers=0,
    )
    
    model_old = SpikingUSegNet(use_attention=False).to(device)
    ckpt_old = torch.load("experiments/brats23/fold0/axial_slice/best_model.pt", map_location=device)
    state_dict_old = {k.replace("module.", ""): v for k, v in ckpt_old["model_state_dict"].items()}
    model_old.load_state_dict(state_dict_old)
    model_old.eval()
    
    model_new = SpikingUSegNet(use_attention=True).to(device)
    ckpt_new = torch.load("experiments/brats23/fold0/2_5d_attention/best_model.pt", map_location=device)
    state_dict_new = {k.replace("module.", ""): v for k, v in ckpt_new["model_state_dict"].items()}
    model_new.load_state_dict(state_dict_new)
    model_new.eval()
    
    out_dir = Path("outputs/visualizations")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            target = batch["target"].to(device)
            cid = batch["case_id"][0]
            slice_idx = batch["slice_idx"][0].item()
            
            with torch.amp.autocast(device_type="cuda"):
                model_new.reset_states()
                for t in range(image.shape[1]):
                    pred_new = model_new(image[:, t])
                
                model_old.reset_states()
                pred_old = model_old(image[:, -1])
                
            pn = (pred_new > 0.5).float()[0].cpu().numpy()
            po = (pred_old > 0.5).float()[0].cpu().numpy()
            gt = target[0].cpu().numpy()
            
            dice_n = [compute_dice(pn[i], gt[i]) for i in range(3)]
            dice_o = [compute_dice(po[i], gt[i]) for i in range(3)]
            
            bg = image[0, -1, 1].cpu().numpy()
            
            gt_map = to_brats_map(gt)
            pn_map = to_brats_map(pn)
            po_map = to_brats_map(po)
            
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            fig.patch.set_facecolor("#111111")
            
            axes[0].imshow(bg.T, cmap="gray", origin="lower", vmin=0, vmax=1)
            axes[0].set_title("Raw T1c", color="white", fontsize=12)
            
            axes[1].imshow(bg.T, cmap="gray", origin="lower", vmin=0, vmax=1)
            axes[1].imshow(create_rgba_overlay(gt_map).transpose(1, 0, 2), origin="lower")
            axes[1].set_title("Ground Truth", color="white", fontsize=12)
            
            axes[2].imshow(bg.T, cmap="gray", origin="lower", vmin=0, vmax=1)
            axes[2].imshow(create_rgba_overlay(pn_map).transpose(1, 0, 2), origin="lower")
            axes[2].set_title(f"New Model (2.5D + Attn)\nET:{dice_n[0]:.2f} | TC:{dice_n[1]:.2f} | WT:{dice_n[2]:.2f}", color="white", fontsize=12)
            
            axes[3].imshow(bg.T, cmap="gray", origin="lower", vmin=0, vmax=1)
            axes[3].imshow(create_rgba_overlay(po_map).transpose(1, 0, 2), origin="lower")
            axes[3].set_title(f"Old Model (2D Base)\nET:{dice_o[0]:.2f} | TC:{dice_o[1]:.2f} | WT:{dice_o[2]:.2f}", color="white", fontsize=12)
            
            for ax in axes:
                ax.axis("off")
                ax.set_facecolor("#111111")
                
            legend_elements = [
                Patch(facecolor=_OVERLAY_COLORS[3][:3], label="ET (Enhancing)"),
                Patch(facecolor=_OVERLAY_COLORS[1][:3], label="TC (Core)"),
                Patch(facecolor=_OVERLAY_COLORS[2][:3], label="WT (Edema/Rim)"),
            ]
            fig.legend(handles=legend_elements, loc="lower center", ncol=3, 
                       facecolor="#222222", labelcolor="white", bbox_to_anchor=(0.5, 0.05))
            
            fig.suptitle(f"{cid} | Slice {slice_idx}", color="white", fontsize=16, y=1.05)
            
            save_path = out_dir / f"{cid}_combined_compare.png"
            plt.savefig(str(save_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"Saved {save_path}")

if __name__ == "__main__":
    main()
