# Deviations from Paper (arXiv:2601.16652v1)

This file documents all assumptions made during implementation where the paper
is ambiguous or incomplete. Each item notes what we assumed and why.

---

## Architecture

### D1 — Channel widths at each U-Net stage
**Paper says:** Fig. 1 shows partial numbers `4 → 32 → 64 → 128 → 128`.
**Assumption:** Decoder mirrors encoder exactly: `128 → 128 → 64 → 32`, then a
1×1 conv to 3 output channels (ET, TC, WT). Final output is real-valued (no
PLIF on the last layer — consistent with "integrator" language in the paper).

### D2 — Axial view slice count
**Paper says:** Not stated explicitly.
**Assumption:** After central crop (240×240×155 → 160×192×152), the axial axis
(z) has 152 slices. This is the T used for the axial view.

### D3 — PLIF threshold θ and initial λ
**Paper says:** Not given numerically.
**Assumption:** θ = 1.0 (standard LIF threshold); τ_param initialized so
that λ = sigmoid(0) = 0.5, i.e., τ_param initialized to 0. Tunable via val Dice.

### D4 — GroupNorm group count
**Paper says:** "GroupNorm" is used but group count not specified.
**Assumption:** 8 groups (common default; falls back to min(8, channels) for
channels < 8).

---

## Training

### D5 — Per-class loss weighting
**Paper says:** Hybrid BCE+Dice loss over 3 heads (ET, TC, WT), but class
weighting not specified.
**Assumption:** Equal weighting — unweighted mean of the 3-head losses.

### D6 — Spike-rate-aware FLOP counting convention
**Paper says:** ~87% FLOPs reduction vs. DNN baseline, but exact formula not given.
**Assumption:** Effective FLOPs = dense_flops × mean_spike_rate, following
the "synaptic operations" convention from Yin/Corradi/Bohté (NMI 2023).

---

## Data

### D7 — BraTS 2017 not used
**Paper:** Experiments on both BraTS 2017 and BraTS 2023.
**This implementation:** BraTS 2023 only (BraTS 2017 data not acquired).
Config stubs for BraTS 2017 are included but not exercised.

### D8 — Min–max normalization scope
**Paper says:** "min–max normalize each modality × each view independently."
**Assumption:** Normalization is done per-volume (per subject × per modality),
not globally across the dataset, to match standard BraTS practice.

---

*Last updated: 2026-08-16*
