# Spiking U-Seg-Net

Reproduction of **"Reliable Brain Tumor Segmentation Based on Spiking Neural Networks with Efficient Training"** (Ghiardelli, Tang, Sun — Maastricht University, 2026, arXiv:2601.16652v1).

## Task

3D brain tumor segmentation (ET / TC / WT) from multi-modal MRI using:
- **Spiking U-Seg-Net** — U-Net-style encoder-decoder with PLIF spiking neurons
- **FPTT** (Forward Propagation Through Time) — no BPTT, O(1) memory in sequence length
- **Multi-view ensemble** (sagittal + coronal + axial), voxel-wise probability averaging
- **BraTS 2023 dataset** — 1,251 labeled training cases (4 MRI modalities each)

## Repository Structure

```
spiking-useg-net/
├── configs/                  # YAML hyperparameter configs
├── data/
│   ├── raw/                  # extracted BraTS NIfTI files (gitignored)
│   ├── processed/            # cropped + normalized .npy volumes
│   └── splits/               # 5-fold subject-id lists (json)
├── src/spiking_useg/
│   ├── data/                 # preprocess, slicing, dataset, augmentation
│   ├── models/               # PLIF neurons, surrogate gradient, U-Net
│   ├── training/             # FPTT optimizer, loss, train loop
│   ├── inference/            # per-view prediction, multi-view ensemble
│   ├── eval/                 # Dice, NLL metrics, table report generation
│   ├── efficiency/           # spike-rate-aware FLOPs counter
│   └── utils/                # seed, logging, visualization
├── scripts/                  # CLI entry points
├── tests/                    # pytest unit tests
├── notebooks/                # Jupyter sanity checks
└── experiments/              # checkpoints + metrics (gitignored)
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Preprocess data (extracts ZIP, crops, normalizes, slices, creates splits)

```bash
# Process all 1,251 subjects (takes a while — ~30 GB output)
python scripts/run_preprocess.py --data_dir data --subset 0

# Quick smoke test on 5 subjects only
python scripts/run_preprocess.py --data_dir data --subset 5
```

### 3. Train a single view / fold

```bash
python scripts/train_single_view.py \
    --config configs/brats23.yaml \
    --view axial --fold 0 \
    --max_subjects 5 --epochs 3   # smoke test
```

### 4. Full training (all 3 views × 5 folds)

```bash
python scripts/train_all_folds.py --config configs/brats23.yaml
```

### 5. Ensemble evaluation

```bash
python scripts/run_ensemble_eval.py --config configs/brats23.yaml --fold 0
```

### 6. Run tests

```bash
pytest tests/ -v
```

## Deviations from Paper

See [DEVIATIONS.md](DEVIATIONS.md) for all documented assumptions and deviations.

## Citation

```
Ghiardelli et al. (2026). "Reliable Brain Tumor Segmentation Based on
Spiking Neural Networks with Efficient Training." arXiv:2601.16652v1.

Baid et al. (2021). "The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor
Segmentation and Radiogenomic Classification." arXiv:2107.02314.
```
