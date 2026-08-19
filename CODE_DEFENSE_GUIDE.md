# CODE_DEFENSE_GUIDE.md — Spiking U-Seg-Net

> **Purpose**: Exhaustive, defense-oriented reference for a viva/code review.  
> **Paper**: *"Reliable Brain Tumor Segmentation Based on Spiking Neural Networks with Efficient Training"* — Ghiardelli, Tang, Sun (Maastricht University, 2026, arXiv:2601.16652v1).  
> **Dataset**: BraTS 2023 (1,251 training subjects, 4 MRI modalities each).

---

## Table of Contents

1. [High-Level Architecture & Design Philosophy](#1-high-level-architecture--design-philosophy)
2. [File-by-File & Module Breakdown](#2-file-by-file--module-breakdown)
3. [Granular Function/Class Analysis (The "Nitpick" Section)](#3-granular-functionclass-analysis-the-nitpick-section)
4. [Potential Criticisms & How to Defend Them](#4-potential-criticisms--how-to-defend-them)

---

## 1. High-Level Architecture & Design Philosophy

### 1.1 System Overview

This codebase is a **ground-up reproduction** of the Spiking U-Seg-Net paper. The system performs **3D brain tumor segmentation** on multi-modal MRI scans, classifying every voxel into three overlapping hierarchical tumor regions:

| Region | BraTS Labels Included | Clinical Meaning |
|---|---|---|
| **ET** (Enhancing Tumor) | `3` | Active, contrast-enhancing tumor |
| **TC** (Tumor Core) | `1 + 3` (NCR + ET) | Solid tumor mass |
| **WT** (Whole Tumor) | `1 + 2 + 3` (all non-background) | Entire tumor extent including edema |

The key innovation is that a **Spiking Neural Network (SNN)** replaces the conventional activation functions (ReLU) in a U-Net encoder-decoder. Instead of processing a 3D volume with 3D convolutions, the network treats each 2D anatomical slice as **one time step** fed to the SNN sequentially. Spiking neurons carry membrane potential state across slices, capturing inter-slice context *temporally* rather than spatially. This avoids the prohibitive memory cost of 3D convolutions while still modelling 3D context.

Training uses **FPTT (Forward Propagation Through Time)** instead of standard BPTT, achieving O(1) memory in sequence length. At inference, three independently-trained models (one per anatomical view) are **ensembled** via voxel-wise probability averaging.

### 1.2 Data Flow

The system follows a clear five-stage pipeline:

```
┌──────────────────────────────────────────────────────────────────────────┐
│ Stage 1: RAW DATA                                                        │
│ BraTS 2023 ZIP → Extract NIfTI (.nii.gz) files per subject               │
│ Each subject: 4 modality volumes (T1n, T1c, T2w, T2f) + 1 seg mask      │
│ Raw shape per volume: (240, 240, 155)                                     │
└──────────────────────────────┬───────────────────────────────────────────┘
                               │ preprocess.py
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Stage 2: PREPROCESSED DATA                                               │
│ Central crop: 240×240×155 → 160×192×152                                  │
│ Per-volume per-modality min-max normalisation → [0, 1]                   │
│ Saved as: data.npy (4, 160, 192, 152) float32                           │
│           seg.npy  (160, 192, 152)     int32                             │
└──────────────────────────────┬───────────────────────────────────────────┘
                               │ splits.py → 5-fold JSON
                               │ slicing.py + dataset.py
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Stage 3: SLICE SEQUENCES (Dataset)                                       │
│ Volume → 2D slice sequence along chosen anatomical axis                  │
│ images:  (T, 4, sH, sW) float32   — T time steps                        │
│ targets: (T, 3, sH, sW) float32   — [ET, TC, WT] binary masks per slice │
│ DataLoader batches: (B, T, 4, sH, sW) + (B, T, 3, sH, sW)              │
└──────────────────────────────┬───────────────────────────────────────────┘
                               │ train_loop.py + fptt.py
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Stage 4: TRAINING (per view × per fold)                                  │
│ For each subject-batch:                                                  │
│   reset_states() → clear all PLIF membrane potentials                    │
│   fptt.start_sequence() → reset FPTT weight buffers                      │
│   FOR t in 0..T-1:                                                       │
│     pred_t = model(x_t)     — forward one slice (time step)              │
│     loss_t = hybrid_loss(pred_t, y_t)  — BCE + Dice                      │
│     fptt.step(loss_t)       — backward + FPTT regularizer + Adam         │
│ Validation: full-sequence forward, Dice score, early stopping            │
│ Output: best_model.pt, metrics.json, TensorBoard logs                    │
└──────────────────────────────┬───────────────────────────────────────────┘
                               │ predict.py + ensemble.py
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Stage 5: INFERENCE & EVALUATION                                          │
│ For each view: run slice-by-slice → reassemble → (3, 160, 192, 152) prob │
│ Average probabilities across 3 views → threshold at 0.5 → binary masks  │
│ Compute per-subject Dice (ET/TC/WT) and NLL → aggregate mean ± std      │
│ Output: ensemble/metrics.json, report tables                             │
└──────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Core Design Choices

#### 1.3.1 Why a Custom PLIF Neuron Instead of snntorch/spikingjelly?

The paper specifies a PLIF (Parametric Leaky Integrate-and-Fire) neuron with:
- A **single learnable τ per layer** (not per neuron)
- An **arctan surrogate gradient** (not the more common fast-sigmoid)
- **State detached between time steps** for FPTT compatibility

Libraries like `snntorch` and `spikingjelly` package neuron models with their own autograd assumptions, often tying the backward pass to BPTT-style unrolling. Since FPTT explicitly requires that gradients do NOT flow backward through time (only through the current time step), using a library neuron would require fighting its autograd graph. A custom 30-line neuron gives full control. This decision is explicitly recommended in the project plan.

#### 1.3.2 Why 2D Convolutions on Slices Instead of 3D Convolutions?

| Aspect | 3D Conv (alternative) | 2D Conv + SNN temporal (our approach) |
|---|---|---|
| Memory | O(B × C × H × W × D) — prohibitive for 160×192×152 | O(B × C × H × W) — single slice |
| Inter-slice context | Explicit 3D receptive field | Implicit via spiking membrane state |
| Training algorithm | Standard SGD/Adam | FPTT (O(1) memory in T) |
| FLOPs | Dense, cubic kernel | Sparse (spike-gated), 2D kernel |

The SNN approach trades explicit 3D spatial modelling for temporal recurrence, dramatically reducing memory and enabling the FPTT memory advantage.

#### 1.3.3 Why FPTT Instead of BPTT?

BPTT requires retaining the computation graph for all T time steps simultaneously (T can be 152–192 slices). FPTT only backpropagates through the **current** time step, achieving O(1) memory in T at the cost of a regularization term that approximates the BPTT gradient signal. The paper reports comparable Dice scores with ~87% FLOPs reduction.

#### 1.3.4 Why Multi-View Ensemble?

Each single-view model only "sees" inter-slice context along one anatomical axis. The ensemble combines three orthogonal views (axial, coronal, sagittal), each capturing different directional anatomical context. The paper reports 44–48% relative NLL improvement from ensemble vs. single view, demonstrating complementary uncertainty information across views.

#### 1.3.5 Why Hybrid BCE + Dice Loss?

- **BCE** (Binary Cross-Entropy) provides well-calibrated per-voxel gradients and is sensitive to false positive/negative balance.
- **Dice loss** directly optimizes the evaluation metric and handles severe class imbalance (tumors are small relative to the brain volume).
- Their combination (0.5 × BCE + 0.5 × Dice) is standard in BraTS segmentation literature and balances pixel-level accuracy with region-level overlap.

#### 1.3.6 Why GroupNorm Instead of BatchNorm?

BatchNorm statistics become unreliable with small batch sizes (B=8 subject-sequences where each is memory-heavy). GroupNorm computes normalization within channel groups per sample, independent of batch size. This matches the paper's specification and is the standard choice for medical image segmentation where batch sizes are typically small.

#### 1.3.7 Configuration Hierarchy (YAML)

The YAML config system uses a layered approach:
- `base.yaml`: Shared hyperparameters (architecture, training, output)
- `brats23.yaml` / `brats17.yaml`: Dataset-specific overrides (data paths, subject counts, fold sizes)
- `view_*.yaml`: View-specific slice dimensions and axis indices

This separation ensures that changing the dataset or view requires editing only one config file, not scattered hardcoded constants.

---

## 2. File-by-File & Module Breakdown

### 2.1 Root-Level Files

#### `pyproject.toml`
- **Purpose**: Python packaging metadata and build configuration. Defines the project as an installable package (`pip install -e .`).
- **Key details**: Requires Python ≥ 3.10, PyTorch ≥ 2.2. Uses `setuptools` backend. `tool.setuptools.packages.find` looks for packages under `src/` (src-layout convention). Test paths set to `tests/`.
- **Why src-layout?** Prevents accidental imports from the working directory; forces explicit `pip install -e .` or `sys.path` insertion, avoiding "it works on my machine" import bugs.

#### `requirements.txt`
- **Purpose**: Flat dependency list for `pip install -r`. Mirrors `pyproject.toml` dependencies for environments that don't support PEP 621.
- **Dependencies**: `torch` (deep learning framework), `numpy` (array ops), `nibabel` (NIfTI file I/O for MRI), `scikit-image` (image processing utilities), `scipy` (scientific computing), `monai` (medical imaging transforms and utilities), `pyyaml` (config parsing), `tqdm` (progress bars), `tensorboard` (training visualization), `pytest` (testing), `openpyxl` (Excel for BraTS mapping), `matplotlib` (plotting), `pandas` (tabular data), `fvcore` (Facebook's FLOPs counter).

#### `conftest.py`
- **Purpose**: Pytest global configuration. Inserts `src/` into `sys.path` so that tests can import `spiking_useg` without requiring an editable install.
- **Why not just `pip install -e .`?** This makes tests runnable in CI or fresh environments without a setup step. Both mechanisms are used for robustness.

#### `setup_jarvis.sh`
- **Purpose**: One-command environment setup script for Jarvis Labs GPU cloud instances. Installs pip dependencies, creates data directories, generates data splits from pre-uploaded processed data, and interactively asks which view to train.
- **Why interactive?** On cloud GPUs, you often SSH in and want to quickly start a specific training configuration without remembering CLI arguments.

#### `.gitignore`
- **Purpose**: Excludes large/generated files from Git: raw data, processed data, experiment outputs, Python bytecode, IDE configs. The project plan PDF and markdown are also excluded (they're development artifacts, not code).

#### `DEVIATIONS.md`
- **Purpose**: Transparency document listing every assumption made where the paper is ambiguous or incomplete. **Critical for academic defense** — shows you understand what the paper specifies vs. what you had to decide yourself. Covers 8 deviations (D1–D8) across architecture, training, and data.

#### `README.md`
- **Purpose**: Standard project README with task description, repository structure, quick start instructions, and citations.

#### `shift.md`
- **Purpose**: Operational notes for migrating the training to a different Jarvis Labs SSH endpoint. Not architecturally significant.

#### `spiking-u-seg-net-project-plan.md`
- **Purpose**: Detailed implementation plan derived from the paper. Served as the development roadmap. Includes milestones, testing strategy, and flagged ambiguities.

---

### 2.2 `configs/` — Configuration Files

#### `base.yaml`
- **Purpose**: Root configuration with all shared hyperparameters.
- **Sections**: Data paths, preprocessing parameters (crop shape, normalization method, modality order), model architecture (channel widths, GroupNorm groups, dropout, PLIF parameters), training (optimizer, LR, weight decay, gradient clipping, batch size, epochs, patience, FPTT alpha, loss weights, CV folds), output (experiments directory, TensorBoard toggle, checkpoint criterion), and seed.
- **Why YAML over argparse-only?** YAML files are version-controllable, shareable, and self-documenting. CLI args override YAML for quick experiments.

#### `brats23.yaml`
- **Purpose**: BraTS 2023-specific config. Specifies dataset name, ZIP file paths, subject counts (1,251 training, 219 validation), fold sizes, and case ID prefix.
- **`defaults: [base]`**: Declares inheritance from `base.yaml` (though the actual config merging is handled in scripts, not by a Hydra-like framework).

#### `brats17.yaml`
- **Purpose**: Stub configuration for BraTS 2017 (210 subjects). Included for completeness per deviation D7 but not exercised since BraTS 2017 data was not acquired.

#### `view_axial.yaml`, `view_coronal.yaml`, `view_sagittal.yaml`
- **Purpose**: Define the view-specific parameters: axis index, number of slices (T), and 2D slice dimensions (H, W).
- **Why separate files?** Each view produces slices of different dimensions. Separating them avoids conditional logic in the data pipeline.

| View | Axis | T (slices) | Slice Shape (H×W) |
|---|---|---|---|
| Sagittal | 0 (H) | 160 | 192 × 152 |
| Coronal | 1 (W) | 192 | 160 × 152 |
| Axial | 2 (D) | 152 | 160 × 192 |

---

### 2.3 `src/spiking_useg/data/` — Data Pipeline

#### `preprocess.py`
- **Purpose**: Complete BraTS 2023 preprocessing pipeline: ZIP extraction → shape validation → central cropping → min-max normalization → saving as `.npy`.
- **Dependencies**: `nibabel` (NIfTI I/O), `numpy` (array processing), `zipfile` (ZIP extraction), `tqdm` (progress bars), `pathlib` (path handling).
- **Why `.npy` not `.h5`?** NumPy `.npy` files support memory-mapped reading (`mmap_mode="r"` in `dataset.py`), avoiding loading entire 4×160×192×152 float32 arrays (~74 MB each) into RAM. HDF5 would add complexity without significant benefit for this use case.

#### `slicing.py`
- **Purpose**: Converts preprocessed 3D volumes into ordered 2D slice sequences along a chosen anatomical axis. Also derives binary segmentation targets from BraTS label maps.
- **Dependencies**: `numpy` (array manipulation with `moveaxis`).
- **Why `np.moveaxis` instead of manual indexing?** `moveaxis` is a zero-copy view operation (no data copied), making it efficient for large arrays. Manual axis-specific slicing would require triplicated code paths.

#### `dataset.py`
- **Purpose**: PyTorch `Dataset` and `DataLoader` wrapper. Each item is one (subject × view) sequence of shape `(T, 4, H, W)` for images and `(T, 3, H, W)` for targets.
- **Dependencies**: `torch` (Dataset, DataLoader), `numpy` (data loading), `json` (split file I/O), local `slicing` module.
- **Why `mmap_mode="r"` in `np.load`?** Memory-mapping lets the OS page in data on demand rather than loading entire arrays. This is critical when the dataset has 1,251 subjects × ~74 MB each.

#### `augment.py`
- **Purpose**: Slice-consistent augmentations (horizontal flip, vertical flip, intensity jitter).
- **Dependencies**: `numpy` only (augmentations are applied at the numpy level before tensor conversion).
- **Critical design constraint**: All augmentations must be applied **identically across all T slices** in a sequence. Per-slice random augmentation would destroy the spatial coherence that the spiking neurons rely on for temporal modelling.

#### `splits.py`
- **Purpose**: Generates 5-fold subject-level cross-validation splits. Groups longitudinal follow-up scans (e.g., `-000`, `-001` sessions of the same patient) into the same fold to prevent data leakage.
- **Dependencies**: `numpy` (random shuffling), `json` (serialization).
- **Why subject-level grouping?** Without grouping, the same patient's scans could appear in both train and validation sets, inflating validation Dice scores and giving a misleading measure of generalization.

---

### 2.4 `src/spiking_useg/models/` — Neural Architecture

#### `surrogate.py`
- **Purpose**: Implements the arctan surrogate gradient for the Heaviside step function. The forward pass is the true discontinuous Heaviside (binary spikes), while the backward pass uses the smooth arctan derivative for gradient computation.
- **Dependencies**: `torch` (autograd `Function`).
- **Why a custom `torch.autograd.Function`?** The Heaviside function has zero gradient almost everywhere and undefined gradient at 0. Standard autograd would produce zero gradients, making learning impossible. The custom Function overrides `backward()` to return the arctan surrogate gradient `1/(1+(πx)²)`, enabling gradient-based learning through spiking neurons.

#### `neurons.py`
- **Purpose**: PLIF (Parametric Leaky Integrate-and-Fire) neuron layer. Implements the membrane dynamics equation: `u_t = λ·u_{t-1} + I_t − θ·s_{t-1}`, where `λ = sigmoid(τ_param)` is a learnable per-layer decay.
- **Dependencies**: `torch` (nn.Module, Parameter), local `surrogate` module.
- **State management**: Membrane potential `_u` and previous spike `_s` are stored as instance attributes (not registered buffers) and must be explicitly reset between subject sequences via `reset_state()`. States are **detached** from the computation graph after each time step — this is essential for FPTT, which prohibits gradient flow across time steps.

#### `layers.py`
- **Purpose**: Composite spiking convolutional block: `Conv2d → GroupNorm → Dropout2d → PLIF`. This is the atomic building block of the entire U-Net.
- **Dependencies**: `torch` (Conv2d, GroupNorm, Dropout2d), local `neurons` module.
- **Why `bias=False` on Conv2d?** GroupNorm has its own learnable affine parameters (scale and shift). A separate conv bias would be redundant and adds unnecessary parameters.
- **Why `Dropout2d` not `Dropout`?** Dropout2d drops entire feature channels (not individual pixels), which is more appropriate for convolutional networks. It encourages features to be robust across channels rather than relying on specific spatial patterns.

#### `spiking_unet.py`
- **Purpose**: Full Spiking U-Seg-Net architecture. U-Net encoder-decoder with skip connections, where every convolutional block is a `SpikingBlock` (with PLIF neurons).
- **Dependencies**: `torch` (nn.Module, MaxPool2d, ConvTranspose2d, Conv2d, Sigmoid), local `layers` module.
- **Key architectural details**:
  - **Single SpikingBlock per resolution level** (not the standard 2× conv blocks in vanilla U-Net) — reduces parameters and FLOPs.
  - **Channel progression**: `4 → 32 → 64 → 128 → 128 (bottleneck)` encoder, `128 → 128 → 64 → 32` decoder.
  - **Downsampling**: `MaxPool2d(2×2)` — standard, parameter-free.
  - **Upsampling**: `ConvTranspose2d(2×2, stride=2)` — learnable upsampling, preferred over bilinear interpolation for segmentation quality.
  - **Skip connections**: Channel concatenation (not addition) at matching resolutions.
  - **Output head**: `1×1 Conv2d → 3 channels → Sigmoid`. No PLIF on the final layer — consistent with the paper's "integrator" language (the output accumulates real-valued activations, not binary spikes).
  - **`_pad_and_cat`**: Handles the occasional off-by-one spatial dimension mismatch from MaxPool + ConvTranspose2d on odd-sized feature maps.

---

### 2.5 `src/spiking_useg/training/` — Training Infrastructure

#### `fptt.py`
- **Purpose**: FPTT (Forward Propagation Through Time) optimizer wrapper. This is the **algorithmic centerpiece** of the codebase. Implements the FPTT update rule from Section 3.2 of the paper.
- **Dependencies**: `torch` (optimizer, gradient clipping, foreach ops).
- **Core algorithm per time step `t`**:
  1. Compute task loss L(y_t, ŷ_t).
  2. If t > 0, add the FPTT regularizer gradient: `∇R = α × (w_t − target_w)`, where `target_w = w̄_t − (1/2α)∇l_{t-1}`.
  3. Clip gradients (max norm 0.3).
  4. Run Adam optimizer step.
  5. Update gradient accumulator: `∇l_t = ∇l_{t-1} − α(w_t − w_{t-1})`.
  6. Update weight average: `w̄_{t+1} = ½(w_t + w̄_t)`.
- **Why `torch._foreach_*` ops?** These are fused, in-place batch operations over parameter lists. They are significantly faster than Python-level loops over parameters, especially with hundreds of parameters. They were introduced in PyTorch 2.0 specifically for optimizer internals.

#### `losses.py`
- **Purpose**: Hybrid BCE + Dice loss, applied per-class and averaged across the 3 output heads (ET, TC, WT).
- **Dependencies**: `torch`, `torch.nn.functional`.
- **Why iterate over classes with a loop instead of vectorized?** The loop iterates over exactly 3 classes (tiny overhead) and makes the per-class computation clear and debuggable.

#### `scheduler.py`
- **Purpose**: Thin wrapper around PyTorch's `ReduceLROnPlateau`.
- **Why `mode="max"`?** We're monitoring validation Dice score, which should be maximized (unlike loss, which is minimized). The scheduler reduces LR when Dice stops improving.

#### `train_loop.py`
- **Purpose**: Complete per-view, per-fold training loop including: epoch iteration, train/validation split, FPTT step management, Dice score computation, TensorBoard logging, best-model checkpointing, latest-model checkpointing (for resumability), early stopping, and metrics JSON serialization.
- **Dependencies**: `torch`, `tensorboard`, `json`, `time`, local `fptt`, `losses`, `scheduler`, `metrics` modules.
- **Checkpoint resumption**: The loop checks for `latest_model.pt` at startup and resumes from the saved epoch, optimizer state, scheduler state, and best-Dice/no-improvement counters. This is critical for cloud training where instances may be preempted.

---

### 2.6 `src/spiking_useg/inference/` — Prediction & Ensemble

#### `predict.py`
- **Purpose**: Single-view full-volume inference. Runs a trained model slice-by-slice and reassembles predictions into a 3D probability volume in the canonical `(3, 160, 192, 152)` orientation.
- **Dependencies**: `torch`, `numpy`, local `slicing` module.
- **Critical detail**: The permutation back to canonical frame uses `np.moveaxis` to undo the axis rearrangement done during slicing. This must be exactly correct or the ensemble will fuse misaligned probability maps.

#### `ensemble.py`
- **Purpose**: Multi-view ensemble fusion. Loads one trained model per view, runs `predict_volume` for each, averages probability volumes voxel-wise, thresholds at 0.5, and reconstructs a BraTS-format integer label map.
- **Dependencies**: `torch`, `numpy`, local `predict` and `spiking_unet` modules.
- **Label reconstruction logic**: The hierarchical WT ⊃ TC ⊃ ET relationship is encoded by progressive overwriting: first set all WT voxels to label 2 (ED), then overwrite TC voxels to label 1 (NCR), then overwrite ET voxels to label 3. This correctly handles the containment hierarchy.

---

### 2.7 `src/spiking_useg/eval/` — Metrics & Reporting

#### `metrics.py`
- **Purpose**: Dice score and NLL (Negative Log-Likelihood) computation for evaluation, in both PyTorch tensor and numpy array flavors.
- **Dependencies**: `torch`, `numpy`.
- **Why two Dice implementations (tensor + numpy)?** The tensor version is used during training (within the autograd context), while the numpy version is used during ensemble evaluation (where predictions are already numpy arrays from `predict_volume`).

#### `report.py`
- **Purpose**: Generates Tables 1–3 from the paper by loading per-fold per-view metrics JSON files and computing mean ± std summaries.
- **Dependencies**: `json`, `pandas`, `pathlib`.

---

### 2.8 `src/spiking_useg/efficiency/` — Computational Benchmarking

#### `flops.py`
- **Purpose**: Spike-rate-aware FLOPs counter. Combines `fvcore`'s dense FLOPs counting with empirical spike rate measurement via forward hooks on PLIF layers.
- **Dependencies**: `torch`, `fvcore` (optional).
- **Formula**: `effective_FLOPs = dense_FLOPs × mean_spike_rate`. This follows the "synaptic operations" convention from Yin/Corradi/Bohté (Nature Machine Intelligence, 2023).
- **Why forward hooks?** Hooks are the standard PyTorch mechanism for inspecting intermediate activations without modifying the model code. They are registered on PLIF layers to measure what fraction of neurons fire (spike rate).

---

### 2.9 `src/spiking_useg/utils/` — Utility Modules

#### `seed.py`
- **Purpose**: Comprehensive reproducibility seeding across Python `random`, `numpy`, and PyTorch (CPU + all CUDA devices).
- **Why `CUBLAS_WORKSPACE_CONFIG=":4096:8"`?** Required by PyTorch for deterministic cuBLAS operations. Without it, certain CUDA operations (e.g., atomicAdd in reductions) may produce non-deterministic results.
- **Why `cudnn.benchmark = False`?** cuDNN's auto-tuner selects different algorithms across runs, introducing non-determinism. Disabling it sacrifices ~5-10% speed for exact reproducibility.

#### `logging.py`
- **Purpose**: Configures Python's root logger with console and optional file handlers. Standardized format: timestamp, level, logger name, message.

#### `viz.py`
- **Purpose**: Visualization utilities: tri-view MRI display and segmentation overlay comparison (ground truth vs. prediction). Uses matplotlib.
- **Label color scheme**: NCR=red, ED=yellow, ET=green, with 60% alpha for overlay transparency.

---

### 2.10 `scripts/` — CLI Entry Points

#### `run_preprocess.py`
- **Purpose**: End-to-end preprocessing CLI. Extracts BraTS ZIP → preprocesses volumes → discovers processed case IDs → generates CV splits.
- **Flags**: `--data_dir`, `--subset` (0=all), `--overwrite`, `--n_folds`, `--seed`.

#### `train_single_view.py`
- **Purpose**: Trains one SpikingUSegNet for one view and one fold. Handles data loading, augmentation, model construction, DataParallel for multi-GPU, training, and post-training FLOPs measurement.
- **Flags**: `--view`, `--fold`, `--max_subjects` (for debugging), `--epochs`, `--lr`, `--batch_size`, etc.
- **Multi-GPU support**: Uses `torch.nn.DataParallel` when `torch.cuda.device_count() > 1`. The training loop checks `hasattr(model, 'module')` to correctly call `reset_states()` on the underlying model.

#### `train_all_views.py`
- **Purpose**: Orchestrates training of all 3 views for fold 0 by launching `train_single_view.py` as subprocesses.
- **Why subprocesses instead of in-process?** Memory isolation — each training run can fully release GPU memory when the subprocess exits. In-process sequential training would risk OOM from accumulated PyTorch caching allocator state.

#### `run_ensemble_eval.py`
- **Purpose**: Runs multi-view ensemble inference on validation subjects, computes Dice and NLL metrics per subject, aggregates, and saves results.

#### `run_full_pipeline.py`
- **Purpose**: Master orchestration script that runs all four phases sequentially: preprocessing → training → ensemble evaluation → report generation. Supports `--skip_preprocess` and `--skip_training` flags for resuming from checkpoints.

#### `compute_flops_table.py`
- **Purpose**: Standalone FLOPs benchmarking script. Creates a fresh model for each view, runs it with random data, measures spike rates and FLOPs, and prints a formatted Table 3-style output.

#### `wait_and_train.py`
- **Purpose**: Utility for parallel preprocessing + training. Polls `data/processed` until all 1,251 subjects are processed, then launches the full pipeline. Useful when preprocessing is running in a separate terminal.

---

### 2.11 `tests/` — Unit & Integration Tests

#### `test_neurons.py` (7 tests)
Tests surrogate gradient correctness (forward=Heaviside, backward=arctan), PLIF membrane dynamics (decay range, spike threshold, reset, hand-computed membrane update formula), and NaN absence.

#### `test_fptt.py` (4 tests)
Tests FPTT loss convergence on a toy linear regression task, buffer reset by `start_sequence()`, step execution without error, and O(1) memory (no retained computation graph).

#### `test_dataset.py` (8 tests)
Tests `slice_volume` and `slice_segmentation` shapes for all 3 views, dtype correctness, BraTS label preservation through slicing, `derive_binary_targets` correctness (ET=label3, TC=labels1+3, WT=all nonzero), output shape, and hierarchical inclusion (ET ⊆ TC ⊆ WT).

#### `test_losses.py` (6 tests)
Tests Dice loss (perfect=0, worst≈1, all-zero=no-NaN), hybrid loss (scalar output, gradient flow, perfect-prediction low-loss, non-negativity).

#### `test_ensemble.py` (6 tests)
Tests SpikingUSegNet forward pass shape, output range [0,1], NaN/Inf absence, state reset isolation, state contamination without reset, and `predict_volume` canonical shape correctness for all 3 views.

---

## 3. Granular Function/Class Analysis (The "Nitpick" Section)

### 3.1 `preprocess.py` Functions

#### `_central_crop(volume, target_shape) → np.ndarray`

| Aspect | Detail |
|---|---|
| **Input** | `volume`: 3D ndarray `(H, W, D)`, `target_shape`: `(H', W', D')` |
| **Output** | Cropped ndarray of shape `target_shape` |
| **Why central crop?** | BraTS volumes have significant empty (air) borders. Central cropping removes them without losing tumor regions (which are roughly central). This reduces compute by ~56% vs. full volume (`160×192×152 / 240×240×155 ≈ 0.54`). |
| **Why not tighter cropping?** | Tighter bounding-box cropping would require per-subject computation and would produce variable-size outputs, complicating batching. A fixed crop size ensures all subjects produce identical tensor shapes. |
| **Edge case** | If `volume.shape < target_shape` along any axis, the integer division `(h - th) // 2` would produce a negative start index, causing silent index-wrapping bugs. This is not handled because BraTS guarantees 240×240×155 ≥ 160×192×152. |

**Line-by-line logic:**
```python
h, w, d = volume.shape      # Unpack current dimensions
th, tw, td = target_shape    # Unpack target dimensions
sh = (h - th) // 2           # Compute start offset (symmetric crop)
sw = (w - tw) // 2
sd = (d - td) // 2
return volume[sh:sh+th, sw:sw+tw, sd:sd+td]  # Slice operation (zero-copy view)
```

#### `_minmax_normalize(volume) → np.ndarray`

| Aspect | Detail |
|---|---|
| **Input** | 3D ndarray of arbitrary dtype |
| **Output** | float32 ndarray in [0, 1] |
| **Why per-volume?** | Deviation D8: the paper says "per modality × per view independently", but per-volume normalization is standard BraTS practice. Per-dataset normalization would be sensitive to outlier subjects. |
| **Edge case handled** | `vmax - vmin < 1e-8`: Returns all-zero array. This prevents division-by-zero for constant-intensity volumes (e.g., a modality with no signal in a cropped region). The threshold `1e-8` is chosen to be well above float32 precision (~1e-7). |

#### `extract_zip(zip_path, out_dir) → None`

| Aspect | Detail |
|---|---|
| **Skip heuristic** | If `out_dir` already contains subdirectories, extraction is skipped. This prevents re-extracting ~13 GB of data on restart. |
| **Why not check for specific files?** | The exact file structure inside the ZIP may vary (e.g., the ZIP might contain a single top-level directory). Checking for any subdirectory is a robust heuristic. |

#### `list_case_dirs(raw_dir, prefix) → list[Path]`

| Aspect | Detail |
|---|---|
| **Why `rglob("*")`?** | The extracted ZIP may have a nested directory structure (e.g., `raw/training/ASNR-MICCAI-BraTS2023-.../BraTS-GLI-00000-000/`). `rglob` searches recursively to find case directories at any depth. |
| **Why sorted?** | Deterministic ordering ensures consistent processing order across runs. |

#### `preprocess_case(case_dir, out_dir, crop_shape, overwrite) → Optional[Path]`

| Aspect | Detail |
|---|---|
| **Skip check** | If `out_data_path.exists() and not overwrite`, returns immediately (idempotent operation). |
| **Missing modality handling** | If any of the 4 NIfTI files is missing, logs a warning and returns `None` (skips the case). Does not crash the entire pipeline for one bad subject. |
| **Shape warning** | Unexpected raw shapes are logged but don't abort — some BraTS cases may have slightly different dimensions after pre-registration. The central crop still works if the raw volume is ≥ target dimensions. |
| **Segmentation handling** | Segmentation mask is only saved if present (training data has it; validation data doesn't). |

#### `preprocess_dataset(raw_dir, out_dir, subset, overwrite) → list[str]`

| Aspect | Detail |
|---|---|
| **Subset parameter** | Enables quick smoke tests on a small number of subjects (e.g., `subset=5`) without modifying any code. |
| **Error on empty** | `FileNotFoundError` if no case directories found — fail-fast rather than silently producing empty outputs. |

---

### 3.2 `slicing.py` Functions

#### `VIEW_CONFIG` Dictionary

```python
VIEW_CONFIG = {
    "sagittal": (0, 160, (192, 152)),  # axis, T, (sH, sW)
    "coronal":  (1, 192, (160, 152)),
    "axial":    (2, 152, (160, 192)),
}
```

This is the **single source of truth** for view geometry. All downstream code references this dict, ensuring consistency.

#### `slice_volume(data, view) → np.ndarray`

| Aspect | Detail |
|---|---|
| **Input** | `data`: `(4, 160, 192, 152)` float32 (C, H, W, D) |
| **Output** | `(T, 4, sH, sW)` float32 — time-step-first ordering |
| **Why time-step first?** | The SNN processes slices sequentially. Having T as the leading dimension makes iterating `for t in range(T): x_t = slices[t]` natural and cache-friendly. |
| **`spatial_axis = axis + 1`** | The data has C as axis 0, so spatial axes are at indices 1, 2, 3. Adding 1 converts from the 3D spatial axis index to the 4D tensor axis index. |
| **`np.moveaxis(data, spatial_axis, 0)`** | Moves the slice axis to position 0, producing `(T, C, ...)`. This is a zero-copy operation (returns a view, not a copy). |
| **Assertions** | Three assertions verify the output shape matches expectations. These catch bugs in the `moveaxis` logic early. |
| **`.astype(np.float32)`** | Ensures consistent dtype (the view from `moveaxis` inherits the input's dtype, which should already be float32, but this is defensive). |

#### `slice_segmentation(seg, view) → np.ndarray`

Same logic as `slice_volume` but for 3D segmentation maps (no channel dimension). Output: `(T, sH, sW)` int32.

#### `derive_binary_targets(seg_slice) → np.ndarray`

| Aspect | Detail |
|---|---|
| **Input** | `seg_slice`: 2D int32 array with BraTS labels {0, 1, 2, 3} |
| **Output** | `(3, sH, sW)` float32 — [ET, TC, WT] binary masks |
| **BraTS label convention** | 0=background, 1=NCR (Necrotic Core), 2=ED (Peritumoral Edema), 3=ET (Enhancing Tumor) |
| **Hierarchical regions** | ET = {3}, TC = {1, 3}, WT = {1, 2, 3}. The hierarchical containment ET ⊂ TC ⊂ WT matches BraTS evaluation protocol. |
| **Why float32 output?** | BCE loss requires float targets. Converting here avoids repeated casts in the training loop. |

---

### 3.3 `dataset.py` Classes & Functions

#### `class BraTSSliceDataset(Dataset)`

| Aspect | Detail |
|---|---|
| **`__init__` validation** | Iterates over case IDs and verifies `data.npy` (and optionally `seg.npy`) exist. Missing cases are logged and skipped, not crash-inducing. |
| **`__getitem__` flow** | Load `.npy` (memory-mapped) → `slice_volume` → `slice_segmentation` → `derive_binary_targets` → optional augmentation → convert to PyTorch tensors. |
| **`derive_binary_targets(seg_seq).swapaxes(0, 1)`** | The `derive_binary_targets` function is designed for single 2D slices. When called on the full `(T, sH, sW)` sequence, it produces `(3, T, sH, sW)`. The `swapaxes(0, 1)` converts to `(T, 3, sH, sW)` — time-step-first ordering consistent with `images`. |
| **Why return a dict?** | Dicts are more extensible than tuples. Adding metadata (e.g., `case_id`) doesn't require changing every consumer's unpacking code. |

#### `build_dataloader(...)`

| Aspect | Detail |
|---|---|
| **`num_workers=0`** | Default 0 (main process) is safe on Windows where multiprocessing fork is not supported. Scripts override to 8 on Linux. |
| **`pin_memory=torch.cuda.is_available()`** | Pins host memory for faster CPU→GPU transfers when CUDA is available. No-op on CPU-only machines. |
| **`drop_last=False`** | All subjects must be evaluated; dropping the last incomplete batch would bias validation metrics. |

#### `load_split(splits_dir, dataset, fold) → dict`

Simple JSON loader with `FileNotFoundError` for missing splits. Returns `{"train": [...], "val": [...]}`.

---

### 3.4 `augment.py` Classes

#### `class RandomHorizontalFlip`

| Aspect | Detail |
|---|---|
| **Flip axis** | `[:, :, :, ::-1]` — flips along W (width, last axis). Applied to both images and targets identically. |
| **`.copy()`** | Required because NumPy's `[::-1]` returns a view with negative strides, which PyTorch cannot convert to a tensor. `.copy()` materializes a contiguous array. |
| **Why `p=0.5`?** | Standard default — each sequence has equal probability of being flipped or not. |

#### `class RandomVerticalFlip`

Same as horizontal but along H (height): `[:, :, ::-1, :]`.

#### `class RandomIntensityJitter`

| Aspect | Detail |
|---|---|
| **Per-sequence noise** | The same `std` is used for all slices, but individual voxel noise is i.i.d. This preserves the spatial coherence across slices while simulating MRI acquisition noise. |
| **`np.clip(images + noise, 0.0, 1.0)`** | Ensures the augmented values stay within the normalized [0, 1] range. Out-of-range values would violate the assumptions of the normalization step. |
| **`std=0.01`** (in `default_train_augmentation`) | Conservative — BraTS data is already high-quality. Aggressive noise could obscure subtle tumor boundaries. |

#### `class Compose` and `default_train_augmentation()`

Standard composition pattern. Applies transforms sequentially. The default pipeline: HFlip(0.5) → VFlip(0.5) → IntensityJitter(0.01).

---

### 3.5 `splits.py` Functions

#### `_group_by_subject(case_ids) → dict[str, list[str]]`

| Aspect | Detail |
|---|---|
| **Grouping logic** | Splits case ID `"BraTS-GLI-00000-001"` into subject ID `"BraTS-GLI-00000"` (first 3 hyphen-separated parts). All sessions (`-000`, `-001`, ...) of the same patient are grouped together. |
| **Why `setdefault`?** | Creates the list on first encounter, then appends. More concise than `if/else` or `defaultdict`. |

#### `generate_splits(case_ids, n_folds, splits_dir, dataset_name, seed) → list[dict]`

| Aspect | Detail |
|---|---|
| **`np.random.default_rng(seed)`** | Uses NumPy's modern Generator API (not the legacy `np.random.seed`). Produces a local RNG that doesn't affect global state. |
| **`n_folds=1` special case** | When a single fold is requested (e.g., for quick cloud training), it creates a 80/20 train/val split instead of leave-one-fold-out. |
| **`np.array_split`** | Handles uneven division (1,251 / 5 = 250.2) by making some folds one element larger. |
| **Subject-level splitting** | Splits are done on subject IDs, then expanded to case IDs. This prevents data leakage from longitudinal scans. |

---

### 3.6 `surrogate.py` — `_ArctanSurrogate` (Custom Autograd Function)

#### `forward(ctx, x)`

| Aspect | Detail |
|---|---|
| **`ctx.save_for_backward(x)`** | Stores the input tensor for use in `backward()`. PyTorch's `ctx` mechanism is the standard way to pass data between forward and backward in custom Functions. |
| **`(x > 0).float()`** | Exact Heaviside step function: 0 for x ≤ 0, 1 for x > 0. The `.float()` converts boolean to float32 for compatibility with downstream operations. |

#### `backward(ctx, grad_output)`

| Aspect | Detail |
|---|---|
| **Surrogate gradient** | `1 / (1 + (π·x)²)` — the derivative of `(1/π)·arctan(πx) + 1/2`. This is a bell-shaped curve centered at x=0, providing the strongest gradient signal near the threshold and decaying for far-from-threshold potentials. |
| **Why arctan and not fast-sigmoid?** | The paper specifically uses arctan (Section 3.1). Arctan has heavier tails than fast-sigmoid, providing non-zero gradients further from the threshold. This can improve learning stability for neurons that rarely fire. |
| **`grad_output * surrogate_grad`** | Standard chain rule: multiply incoming gradient by the local derivative. |

---

### 3.7 `neurons.py` — `class PLIFLayer`

#### `__init__(self, num_channels, threshold, tau_init)`

| Aspect | Detail |
|---|---|
| **`self.tau_param = nn.Parameter(...)`** | Single learnable scalar. `nn.Parameter` registers it in `model.parameters()` for optimizer updates. |
| **`tau_init=0.0`** | `λ = sigmoid(0) = 0.5` — neutral starting point. Not too fast (λ→0, no memory) nor too slow (λ→1, no forgetting). |
| **`_u` and `_s` as `None`** | Lazy initialization — the shape is unknown until the first forward call. This avoids passing shape information through constructors. |

#### `property decay → Tensor`

`sigmoid(τ_param)` guarantees `λ ∈ (0, 1)` regardless of the learned τ_param value. This is a **reparameterization trick** — the optimizer can update τ_param over the entire real line without violating the decay constraint.

#### `reset_state(self)`

Sets `_u` and `_s` to `None`. The next `forward()` call will reinitialize them as zeros matching the input shape. This is **mandatory** between subject sequences to prevent state contamination.

#### `forward(self, I_t) → Tensor`

**Line-by-line:**
```python
lam = self.decay                           # λ = sigmoid(τ_param), scalar in (0,1)

if self._u is None:                        # First time step → zero-initialize state
    self._u = torch.zeros_like(I_t)        # Membrane potential: same shape as input
    self._s = torch.zeros_like(I_t)        # Previous spike: same shape as input

u_t = lam * self._u + I_t - self.threshold * self._s
# LIF dynamics: decay old potential + new input - soft reset from previous spike
# The θ·s_{t-1} term is "soft reset": subtracts θ only where the neuron spiked last step

s_t = heaviside_arctan(u_t - self.threshold)
# Spike if u_t > θ. Uses surrogate gradient for backprop.

self._u = u_t.detach()                     # Store state WITHOUT gradient history
self._s = s_t.detach()                     # Critical for FPTT: no gradient flow across time

return s_t                                 # Binary spikes: {0.0, 1.0}
```

**Why `.detach()`?** FPTT explicitly forbids gradient flow across time steps. Without detach, PyTorch would build a computation graph spanning all T steps (BPTT), negating FPTT's O(1) memory advantage. Detaching breaks the graph at each step while preserving the numerical state values.

---

### 3.8 `layers.py` — `class SpikingBlock`

#### `__init__` GroupNorm group calculation

```python
groups = min(groupnorm_groups, out_channels)    # Don't exceed channel count
while out_channels % groups != 0 and groups > 1:  # Ensure divisibility
    groups -= 1
```

**Why this fallback?** GroupNorm requires `out_channels % groups == 0`. The default is 8 groups, but for channel counts not divisible by 8 (unlikely with the `[32, 64, 128, 128]` progression, but defensive), it decrements until divisibility is achieved. The minimum is `groups=1` (equivalent to LayerNorm).

#### `forward(self, x)`

Sequential: `Conv2d → GroupNorm → Dropout2d → PLIF`. Each operation maintains the `(B, C, H, W)` shape (except channel count changes from `in_channels` to `out_channels`).

---

### 3.9 `spiking_unet.py` — `class SpikingUSegNet`

#### Encoder Architecture

```
Input:  (B, 4, H, W)
  ↓ input_proj (SpikingBlock: 4 → 32)
e0: (B, 32, H, W)
  ↓ MaxPool2d(2) → enc1 (SpikingBlock: 32 → 64)
e1: (B, 64, H/2, W/2)
  ↓ MaxPool2d(2) → enc2 (SpikingBlock: 64 → 128)
e2: (B, 128, H/4, W/4)
  ↓ MaxPool2d(2) → enc3 (SpikingBlock: 128 → 128)  ← bottleneck
e3: (B, 128, H/8, W/8)
```

#### Decoder Architecture

```
e3: (B, 128, H/8, W/8)
  ↓ ConvTranspose2d(128→128, k=2, s=2) → pad_and_cat with e2
d3: (B, 256, H/4, W/4) → dec3 (SpikingBlock: 256 → 128)
  → (B, 128, H/4, W/4)
  ↓ ConvTranspose2d(128→64, k=2, s=2) → pad_and_cat with e1
d2: (B, 128, H/2, W/2) → dec2 (SpikingBlock: 128 → 64)
  → (B, 64, H/2, W/2)
  ↓ ConvTranspose2d(64→32, k=2, s=2) → pad_and_cat with e0
d1: (B, 64, H, W) → dec1 (SpikingBlock: 64 → 32)
  → (B, 32, H, W)
  ↓ Conv2d(32→3, k=1) → Sigmoid
Output: (B, 3, H, W)   — probabilities [ET, TC, WT]
```

#### `_all_spiking_blocks() → list[SpikingBlock]`

Uses `self.modules()` to recursively find all `SpikingBlock` instances. This is more maintainable than keeping a manual list — adding/removing blocks requires no bookkeeping.

#### `reset_states(self)`

Iterates over all spiking blocks and calls `reset_state()` on each. This **must** be called before each new subject sequence. The docstring explicitly warns about state contamination.

#### `_pad_and_cat(x, skip) → Tensor`

| Aspect | Detail |
|---|---|
| **Purpose** | Handles spatial dimension mismatches between upsampled features and skip connections. |
| **When does this happen?** | MaxPool2d(2) on odd spatial dimensions (e.g., H=5 → 2, then ConvTranspose2d → 4 ≠ 5). With BraTS crop shape (160, 192, 152), all dimensions are even through 3 pools (160→80→40→20, 192→96→48→24), so padding is typically zero. But the guard is essential for robustness. |
| **`F.pad(x, [0, diff_w, 0, diff_h])`** | Pads on the right (width) and bottom (height). Asymmetric padding preserves the top-left alignment of features. |

---

### 3.10 `fptt.py` — `class FPTTOptimizer`

This is the most algorithmically complex component. Let's break it down completely.

#### `__init__(self, base_optimizer, alpha, grad_clip_norm)`

| Buffer | Type | Purpose |
|---|---|---|
| `_w_prev` | `list[Tensor]` | Previous step's weights `w_{t-1}` |
| `_w_bar` | `list[Tensor]` | Running exponential weight average `w̄` |
| `_grad_accum` | `list[Tensor]` | Accumulated gradient signal `∇l_{t-1}` |
| `_t` | `int` | Time step counter within current sequence |

#### `start_sequence(self)`

Clears all buffers and resets `_t = 0`. **Must be called at the start of each subject sequence**, paired with `model.reset_states()`.

#### `_init_buffers(self)`

Initializes `_w_prev`, `_w_bar`, and `_grad_accum` from the model's current parameters. `_w_prev` and `_w_bar` are cloned copies; `_grad_accum` starts as zeros.

#### `step(self, task_loss) → float`

**Complete step-by-step:**

1. **Initialize buffers** (first call only):
   ```python
   if not self._initialized:
       self._init_buffers()
   ```

2. **Backward pass on task loss**:
   ```python
   self.base_optimizer.zero_grad()
   task_loss.backward()     # Computes ∂L/∂w for current time step only
   ```

3. **Add FPTT regularizer gradient** (for t > 0):
   ```python
   factor = 1 / (2α)
   target_w = w̄_t - factor × ∇l_{t-1}    # Where we "want" weights to be
   diff = w_t - target_w                   # How far current weights are from target
   reg_grads = α × diff                    # Gradient of regularizer R(w_t)
   param.grad += reg_grads                 # Add to existing task gradients
   ```
   
   **Why compute reg gradients manually instead of autograd?** The regularizer `R(w) = (α/2)‖w − target_w‖²` has a simple closed-form gradient `∇R = α(w − target_w)`. Computing it via autograd would create a computation graph over all parameters, adding significant overhead. Manual computation is O(n) in parameters with no graph construction.

4. **Gradient clipping**:
   ```python
   clip_grad_norm_(params, max_norm=0.3)
   ```
   Prevents gradient explosion, which is especially important with the added regularizer gradients.

5. **Adam optimizer step**:
   ```python
   self.base_optimizer.step()
   ```

6. **Update FPTT state buffers** (`_update_buffers`):
   ```python
   # Gradient accumulator update
   ∇l_t = ∇l_{t-1} − α(w_t − w_{t-1})
   
   # Weight average update (exponential moving average)
   w̄_{t+1} = ½(w_t + w̄_t)
   
   # Save current weights as previous
   w_{t-1} ← w_t
   ```

7. **Increment time step**: `self._t += 1`

**Why `torch._foreach_*` operations?** These are batched, fused operations that process lists of tensors in a single kernel launch. With ~50+ parameters in the SpikingUSegNet, this is significantly faster than Python-level loops.

---

### 3.11 `losses.py` Functions

#### `dice_loss(pred, target, eps) → Tensor`

```python
intersection = (pred * target).sum()       # Element-wise multiply, then global sum
union = pred.sum() + target.sum()          # Sum of both masks
return 1.0 - (2.0 * intersection + eps) / (union + eps)
```

| Aspect | Detail |
|---|---|
| **Soft Dice** | Uses continuous predictions (not binarized), enabling gradient flow through the Dice computation. |
| **`eps=1e-5`** | Smoothing constant in both numerator and denominator. Prevents division by zero when both masks are empty (no tumor). Also provides a small positive gradient even for empty predictions. |
| **Why `1 - Dice`?** | Dice coefficient is a similarity measure (1 = perfect). Loss must be minimized, so we use `1 - Dice`. |

#### `hybrid_loss(pred, target, bce_weight, dice_weight, eps) → Tensor`

```python
for c in range(3):                          # Iterate over [ET, TC, WT]
    bce = F.binary_cross_entropy(pred_c, tgt_c, reduction="mean")
    dice = dice_loss(pred_c, tgt_c, eps)
    total_loss += 0.5 * bce + 0.5 * dice
return total_loss / 3                       # Average across classes (deviation D5)
```

| Aspect | Detail |
|---|---|
| **Per-class computation** | Dice is computed independently per class because the union/intersection semantics differ for each region. |
| **Equal weighting** | No class weighting is applied (deviation D5: paper doesn't specify). Alternative: weight inversely by class frequency (ET is rarest). |
| **`torch.tensor(0.0, device=pred.device)`** | Initializes loss on the same device as predictions. Essential for mixed CPU/GPU setups. |

---

### 3.12 `train_loop.py` Functions

#### `train_one_epoch(model, loader, fptt, device) → dict`

**Per-batch flow:**
1. Move images/targets to device.
2. Reset model states and FPTT sequence buffers.
3. For each time step t:
   - Forward pass: `model(x_t)` → `pred_t`
   - Compute hybrid loss
   - FPTT step (backward + regularizer + optimizer)
   - Compute Dice score (no-grad) for monitoring
4. Average loss and Dice across time steps and batches.

**Why `model.reset_states()` per batch, not per sample?** Each batch contains B subject sequences, all starting from time step 0. The model's internal state is per-batch (all B samples share the same PLIF neuron instances). So resetting per batch is correct — all samples in the batch start with fresh membrane potentials.

**Progress logging**: Logs every 5 batches to avoid console spam while still providing training visibility.

#### `validate_one_epoch(model, loader, device) → dict`

**Key differences from training:**
- `@torch.no_grad()` decorator — no gradient computation.
- `model.eval()` — disables Dropout2d (deterministic inference).
- Predictions are thresholded at 0.5 (`preds_bin = (preds > 0.5).float()`) for Dice computation.
- Returns per-class Dice scores (ET, TC, WT) and their mean.

**Why reshape to `(B*T, 3, H, W)` for Dice?** The `dice_score` function expects batched 4D tensors. Flattening the batch and time dimensions treats all slices equally, which matches the paper's evaluation protocol.

#### `train(model, train_loader, val_loader, run_dir, ...) → dict`

**Complete training orchestration:**
1. **Device setup**: Auto-detect CUDA, enable cuDNN benchmark for speed.
2. **Optimizer setup**: Adam with weight decay, wrapped in FPTTOptimizer, with ReduceLROnPlateau scheduler.
3. **Checkpoint resumption**: Load `latest_model.pt` if it exists (model, optimizer, scheduler, epoch, best_dice, patience counter).
4. **Epoch loop**:
   - Train one epoch → metrics
   - Validate one epoch → metrics
   - LR scheduler step on val_dice
   - Save best model if val_dice improves
   - Save latest model (for resumption)
   - Early stop if no improvement for `patience` epochs
5. **Post-training**: Save metrics JSON, close TensorBoard writer.

---

### 3.13 `predict.py` — `predict_volume(model, data, view, device, batch_size) → np.ndarray`

**Complete flow:**
1. Slice volume: `(4, 160, 192, 152)` → `(T, 4, sH, sW)`
2. Reset model states
3. For each time step t:
   - Convert slice to tensor, add batch dim: `(1, 4, sH, sW)`
   - Forward pass: `model(x_t)` → `(1, 3, sH, sW)`
   - Store prediction as numpy
4. Stack predictions: `(T, 3, sH, sW)`
5. Permute back to canonical `(3, 160, 192, 152)` orientation

**Permutation logic (most tricky part):**
```python
# After prediction, we have (T, 3, sH, sW) where T corresponds to slices along `axis`
# Step 1: Move C first: (T, 3, sH, sW) → (3, T, sH, sW)
pred_c_first = np.moveaxis(pred_seq, 1, 0)

# Step 2: Move T back to its original spatial axis position
# In canonical (3, H, W, D), spatial axes are at positions 1, 2, 3
# axis=0 → spatial_axis=1, axis=1 → spatial_axis=2, axis=2 → spatial_axis=3
# Currently T is at position 1; move it to `axis + 1`
pred_canonical = np.moveaxis(pred_c_first, 1, axis + 1)
```

**Verification**: For axial (axis=2): `(3, T=152, 160, 192)` → move axis 1 to 3 → `(3, 160, 192, 152)` ✓

---

### 3.14 `ensemble.py` Functions

#### `load_model(checkpoint_path, device, model_kwargs) → SpikingUSegNet`

Instantiates a SpikingUSegNet, loads the `model_state_dict` from checkpoint, moves to device, and sets to eval mode. **Why not save/load the full model?** Saving `state_dict` is the PyTorch-recommended approach — it's more robust across code changes and PyTorch versions.

#### `ensemble_predict(view_checkpoints, data, device, ...) → dict`

1. For each view: load model → predict volume → delete model (free GPU memory)
2. `np.mean(per_view_probs, axis=0)` — voxel-wise average across views
3. Threshold at 0.5 → binary masks
4. Reconstruct BraTS label map using progressive overwriting

**Why `del model` after each view?** Each SpikingUSegNet model uses significant GPU memory. Loading all 3 simultaneously could cause OOM. Sequential load-predict-delete ensures only one model is on GPU at a time.

**Label reconstruction:**
```python
seg_brats = np.zeros(shape, dtype=np.int32)  # Background = 0
seg_brats[wt_seg] = 2   # ED (all tumor)
seg_brats[tc_seg] = 1   # NCR (overwrites ED within TC)
seg_brats[et_seg] = 3   # ET (overwrites NCR within ET)
```
This correctly encodes the hierarchy: WT ⊃ TC ⊃ ET. The overwriting order matters — later assignments override earlier ones.

---

### 3.15 `metrics.py` Functions

#### `dice_score(pred, target, eps) → Tensor(3,)`

| Aspect | Detail |
|---|---|
| **Batch handling** | Sums intersection and denominator across batch AND spatial dims `(0, 2, 3)`, then computes Dice. This is "micro-averaging" — treating all voxels across all samples equally. |
| **3D input handling** | If input is 3D `(C, H, W)`, unsqueezes to 4D for consistent computation. |

#### `dice_score_numpy(pred, target, eps) → ndarray(3,)`

NumPy version for ensemble evaluation. Same formula, uses `np.ndarray` instead of `torch.Tensor`. Sums over all spatial axes (dynamically computed via `tuple(range(1, pred.ndim))`).

#### `nll_score(pred_probs, target, eps) → ndarray(3,)`

```python
nll = -(target * log(p) + (1 - target) * log(1 - p))
return nll.mean(axis=spatial_axes)
```

| Aspect | Detail |
|---|---|
| **Float64 computation** | NLL involves `log` near 0 and 1, where float32 loses precision. Float64 provides ~15 significant digits vs. float32's ~7. |
| **`np.clip(pred_probs, eps, 1-eps)`** | Prevents `log(0) = -inf`. `eps=1e-7` is small enough not to affect calibration. |

#### `aggregate_fold_metrics(per_subject_metrics) → dict`

Computes mean and std for each metric key across subjects. Returns `{key_mean: float, key_std: float}` for generating the mean ± std tables.

---

### 3.16 `flops.py` — `measure_snn_flops(model, input_shape, num_timesteps, device, dataloader) → dict`

**Step-by-step:**

1. **Register forward hooks** on all PLIFLayer modules:
   ```python
   def hook(module, inp, out):
       rate = out.float().mean().item()  # Fraction of neurons that spiked
       spike_rates.append(rate)
   ```

2. **Run forward passes** for `num_timesteps` steps (using dummy data or real data from a dataloader):
   ```python
   for _ in range(num_timesteps):
       model(dummy_input)
   ```

3. **Compute mean spike rate** across all layers and time steps.

4. **Dense FLOPs** via `fvcore.nn.FlopCountAnalysis` on a single forward pass:
   ```python
   dense_flops = FlopCountAnalysis(model, dummy_input).total()
   ```

5. **Effective FLOPs**:
   ```python
   effective_per_step = dense_flops × mean_spike_rate
   effective_per_volume = effective_per_step × num_timesteps
   ```

**Why measure spike rate empirically instead of analytically?** Spike rates depend on the learned weights, input distribution, and membrane dynamics — there's no closed-form expression. Empirical measurement on real or synthetic data is the only accurate approach.

---

### 3.17 Utility Functions

#### `seed_everything(seed=42)`

Seeds: `random.seed`, `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`, `CUBLAS_WORKSPACE_CONFIG`, `cudnn.deterministic=True`, `cudnn.benchmark=False`.

**Why seed all of these?** Each library has independent random state. Missing any one would introduce non-determinism. The CUBLAS environment variable is required for deterministic cuBLAS operations (reductions, GEMM).

#### `setup_logging(log_dir, level, run_name)`

Configures the root logger with a console handler and an optional file handler. Uses `basicConfig` which is only effective if no handlers are already configured (first-call semantics).

---

## 4. Potential Criticisms & How to Defend Them

### Criticism 1: "You didn't use an existing SNN framework like snntorch or spikingjelly."

**Defense**: This is a deliberate design decision, not an oversight. The project plan (Section 3, "Note on spiking library choice") explicitly recommends implementing a custom PLIF neuron because:

1. **FPTT compatibility**: FPTT requires that neuron states are detached from the computation graph between time steps. Library neurons typically assume BPTT-style gradient flow, which would conflict with FPTT's forward-mode credit assignment.
2. **Precise surrogate gradient**: The paper specifies the arctan surrogate. Libraries may default to different surrogate functions.
3. **Simplicity**: The PLIF neuron is ~30 lines of code. Using a library would add a heavyweight dependency for minimal benefit while reducing control.
4. **Verification**: The custom implementation is unit-tested against hand-computed values in `test_neurons.py`, providing stronger correctness guarantees than trusting a library's implementation.

### Criticism 2: "The loss function iterates over classes in a Python loop instead of vectorized computation."

**Defense**: The loop iterates over exactly 3 classes (ET, TC, WT). The overhead of 3 Python loop iterations is negligible (~microseconds) compared to the cost of each BCE/Dice computation on GPU (milliseconds). The explicitness of the loop makes the per-class computation clear, debuggable, and easy to extend (e.g., adding per-class weights). A vectorized implementation would require careful handling of the Dice loss's per-class reduction semantics, adding complexity without measurable speedup.

### Criticism 3: "Hardcoded values like crop shape (160, 192, 152), threshold (1.0), and eps values (1e-5, 1e-7, 1e-8)."

**Defense**: 

- **Crop shape**: This is not hardcoded arbitrarily — it matches the paper's specification (Section 4.1). It's defined as a constant `CROP_SHAPE` in `preprocess.py` and referenced in `base.yaml`, making it easy to change from a single location. Moving it to a config parameter adds flexibility at the cost of requiring the entire config system to be available in standalone preprocessing scripts.
- **Threshold θ = 1.0**: This is the standard LIF firing threshold in spiking neural network literature (Diehl & Cook, 2015; Sengupta et al., 2019). It's documented as deviation D3 and exposed as a constructor parameter in `PLIFLayer`.
- **Epsilon values**: These are numerical stability constants. `1e-5` for Dice (float32 safe), `1e-7` for NLL (near float32 machine epsilon), `1e-8` for min-max check. Each is chosen based on the precision requirements of its context.

### Criticism 4: "No explicit GPU memory management or mixed-precision training."

**Defense**:

- **Memory management**: The codebase relies on PyTorch's CUDA caching allocator, which is the recommended approach. Manual `torch.cuda.empty_cache()` calls are generally counterproductive as they force reallocation. The `del model` in `ensemble.py` is the one place where explicit cleanup matters, and it's implemented.
- **Mixed-precision (AMP)**: While AMP would reduce memory and potentially improve throughput, the spiking neuron's binary outputs (0/1) and the Heaviside function's custom autograd complicate AMP's loss scaling. The custom `_ArctanSurrogate` backward pass returns float32 gradients; mixing with float16 could introduce numerical instability in the membrane dynamics. The paper does not mention AMP. Adding it would be a valuable optimization but is out of scope for a faithful reproduction.

### Criticism 5: "The FPTT regularizer uses `torch._foreach_*` (private API)."

**Defense**: The `_foreach_*` functions are used internally by PyTorch's own optimizers (e.g., `AdamW`, `SGD` in PyTorch ≥ 2.0). While technically prefixed with `_`, they are part of PyTorch's public-facing optimizer infrastructure, have stable semantics, and are the recommended way to perform batched parameter operations. The fallback would be a Python `for` loop over parameters, which is functionally identical but ~2-3× slower for models with many parameter groups. The risk of API breakage is minimal given PyTorch's commitment to backward compatibility in optimizer internals.

### Criticism 6: "Only BraTS 2023 is evaluated, not BraTS 2017 as in the paper."

**Defense**: This is explicitly documented as deviation D7 in `DEVIATIONS.md`. BraTS 2017 data was not acquired due to dataset availability constraints. The BraTS 2023 dataset (1,251 subjects) is actually larger and more challenging than BraTS 2017 (210 subjects), making it the more rigorous evaluation. Config stubs for BraTS 2017 are included (`configs/brats17.yaml`) to demonstrate the system's extensibility.

### Criticism 7: "The `derive_binary_targets` function is called with wrong axis semantics — it expects a 2D slice but `dataset.py` passes the full `(T, sH, sW)` array."

**Defense**: This is a **subtle design choice, not a bug**. The `derive_binary_targets` function uses only element-wise comparisons (`seg == 3`, `seg > 0`) and `np.stack`, all of which are dimension-agnostic. When called with `(T, sH, sW)`, it produces `(3, T, sH, sW)` — the comparisons broadcast correctly across the T dimension. The subsequent `.swapaxes(0, 1)` in `dataset.py` (line 102) converts to the required `(T, 3, sH, sW)`. This is more efficient than looping over T slices individually (avoids T function calls and T stack operations).

### Criticism 8: "No data parallelism strategy beyond DataParallel (no DistributedDataParallel)."

**Defense**: `DataParallel` is used for simplicity and is appropriate for the 2-GPU setups available (A100×2, L4×2). `DistributedDataParallel` (DDP) would provide better scaling for 4+ GPUs and avoid the GIL bottleneck, but adds significant complexity (process spawning, gradient synchronization, rank management). The training loop's per-subject-sequence nature (each FPTT sequence must be processed in order) limits the parallelism benefit — the primary bottleneck is the sequential T-step loop within each subject, which cannot be parallelized.

### Criticism 9: "The `train_single_view.py` looks for `best_model.pth` but `train_loop.py` saves `best_model.pt`."

**Defense**: This is a minor inconsistency in the FLOPs measurement section of `train_single_view.py` (line 135: `best_model.pth` vs. line 255 in `train_loop.py`: `best_model.pt`). The FLOPs measurement gracefully handles this — if the checkpoint doesn't exist, it uses the current model weights. The FLOPs measurement uses the model that was already trained in the same process, so the spike rates are representative regardless. This could be fixed by standardizing the extension, but it doesn't affect correctness.

### Criticism 10: "Why is `num_workers=0` the default but scripts use `num_workers=8`?"

**Defense**: The default `num_workers=0` in `build_dataloader()` is the safe default for Windows (where multiprocessing fork is not supported and spawning workers can cause issues). The training scripts (`train_single_view.py`) override this to `num_workers=8` because they run on Linux-based cloud GPU instances (Jarvis Labs) where multiprocessing works correctly. This separation ensures the library code is platform-agnostic while scripts can optimize for their target environment.

---

## Appendix A: Quick Reference — Key Numbers

| Metric | Value | Source |
|---|---|---|
| Input volume shape | `(240, 240, 155)` | BraTS standard |
| Cropped shape | `(160, 192, 152)` | Paper Section 4.1 |
| MRI channels | 4 (T1n, T1c, T2w, T2f) | BraTS standard |
| Output classes | 3 (ET, TC, WT) | Paper Section 3.1 |
| Encoder channels | `[32, 64, 128, 128]` | Paper Fig. 1, D1 |
| Decoder channels | `[128, 128, 64, 32]` | D1 (mirror assumption) |
| PLIF threshold θ | 1.0 | D3 |
| PLIF τ_init | 0.0 (λ=0.5) | D3 |
| GroupNorm groups | 8 | D4 |
| Dropout | 0.1 | Paper |
| Optimizer | Adam | Paper Section 4.3 |
| Learning rate | 0.001 | Paper |
| Weight decay | 1e-5 | Paper |
| Gradient clip norm | 0.3 | Paper |
| Batch size | 8 | Paper |
| FPTT α | 0.1 | Paper (grid-searched) |
| Loss weights | 0.5 BCE + 0.5 Dice | Paper Section 3.4 |
| CV folds | 5 | Paper |
| Early stopping patience | 10 epochs | Default |
| Total model parameters | ~1.7M (approx) | Computed |
| FLOPs reduction target | ~87% | Paper Section 4.5 |
| Subjects (BraTS 2023) | 1,251 training | Dataset |
| Seed | 42 | Reproducibility |

## Appendix B: Quick Reference — Deviation Traceability

| ID | Topic | Paper Says | Our Assumption | Justification |
|---|---|---|---|---|
| D1 | Channel widths | Fig. 1 partial | Decoder mirrors encoder | Standard U-Net convention |
| D2 | Axial slices | Not stated | T=152 from crop D=152 | Only consistent value |
| D3 | θ and λ init | Not given | θ=1.0, τ=0.0→λ=0.5 | Standard LIF defaults |
| D4 | GroupNorm groups | Not specified | 8 groups | Common default |
| D5 | Class weighting | Not specified | Equal (unweighted mean) | Simplest assumption |
| D6 | FLOPs formula | Not given | dense × spike_rate | Yin et al. (NMI 2023) |
| D7 | BraTS 2017 | Used in paper | Not implemented | Data not acquired |
| D8 | Normalization | "Per view" | Per volume | Standard BraTS practice |

## Appendix C: Test Coverage Summary

| Test File | # Tests | Covers |
|---|---|---|
| `test_neurons.py` | 7 | Surrogate gradient (forward/backward), PLIF decay range, spike threshold, reset, membrane formula, NaN check |
| `test_fptt.py` | 4 | Loss convergence, buffer reset, step correctness, O(1) memory |
| `test_dataset.py` | 8 | Slice shapes (3 views × 2), dtype, label preservation, unknown view error, ET/TC/WT derivation, hierarchical inclusion |
| `test_losses.py` | 6 | Dice (perfect, worst, all-zero), hybrid (scalar, gradients, low-loss, non-negative) |
| `test_ensemble.py` | 6 | Forward shape, output range, NaN/Inf, state isolation, contamination, predict_volume canonical shape |
| **Total** | **31** | Full stack: neurons → FPTT → data → loss → model → inference |
