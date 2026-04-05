# 🌫️ AISEHack Phase 2 — Theme 2: PM2.5 Concentration Forecasting
### Team HackSprint

---

## 🏆 Competition Overview

**Task**: Predict hourly PM2.5 concentration maps over India (140×124 grid) for 16 hours ahead, given 10 hours of meteorological and pollution history.

**Dataset**: 4 months of 2016 data (April, July, October, December) with 16 atmospheric variables at hourly resolution. Test set: 218 samples from 2017.

**Metric**: Composite Score = `(1 - gSMAPE/2 + (eCorr+1)/2 + 1 - eSMAPE/2) / 3`
- **gSMAPE**: Global Symmetric Mean Absolute Percentage Error
- **eCorr**: Episode Pearson Correlation
- **eSMAPE**: Episode SMAPE (pollution spike events)

---

## 🧠 Our Approach — Multi-Architecture Ensemble

We deploy a **two-model ensemble** combining architecturally diverse models trained with different random seeds, maximizing prediction diversity and reducing generalization error.

### Model A: V15 — TCN → BiConvLSTM → ConvGRU → Channel Attention + UNet

| Component | Details |
|---|---|
| **Temporal Conv Net (TCN)** | 3 dilated Conv3d layers (d=1,2,4) capturing multi-scale temporal patterns from 1h to 4h receptive fields |
| **Bidirectional ConvLSTM** | Forward + backward spatial LSTM cells capturing bidirectional temporal dependencies across the 10-hour lookback window |
| **ConvGRU** | Unidirectional Convolutional GRU for temporal refinement — captures asymmetric time dependencies (past→future) |
| **Channel Attention** | SE-Net style squeeze-excitation block that adaptively reweights feature channels based on their importance |
| **UNet Decoder** | 3-level encoder-decoder with skip connections for multi-scale spatial reconstruction |
| **Residual Baseline** | Last-frame persistence baseline + learned residual correction |
| **Parameters** | ~5M |
| **Seed** | 42 |

### Model B: V17 — STUNet (Spatio-Temporal UNet)

| Component | Details |
|---|---|
| **Frame Encoder** | Shared Conv2d across all timesteps for per-frame spatial feature extraction |
| **Temporal Collapse** | Single Conv3d kernel (10×1×1) fusing all 10 timesteps into a single spatial feature map |
| **UNet Backbone** | 4-level encoder (48→96→192→384 channels) with residual ConvBlocks + 3-level decoder with skip connections |
| **Dropout Regularization** | Dropout2d at bottleneck (0.20) and decoder (0.10) to prevent overfitting |
| **Residual Baseline** | Last-frame persistence + learned residual |
| **Parameters** | ~4.5M |
| **Seed** | 99 |

### Ensemble Strategy

```
Final Prediction = 0.6 × V15_prediction + 0.4 × V17_prediction
```

- **V15 weight = 0.6**: Higher weight due to superior standalone LB score (0.8815)
- **V17 weight = 0.4**: Provides architectural diversity (pure convolutional vs. recurrent)
- **Diversity**: Different architectures (TCN+LSTM vs Conv3d) + different seeds (42 vs 99) = uncorrelated errors

---

## 🔬 Feature Engineering — 13-Channel Input

We engineer **13 informative channels** from 9 raw meteorological variables, dropping 6 zero-valued emission features:

| Channel | Source | Description |
|---|---|---|
| 1 | `cpm25` | PM2.5 concentration (log1p + z-score normalized) |
| 2 | `pblh` | Planetary Boundary Layer Height |
| 3 | `psfc` | Surface Pressure |
| 4 | `q2` | 2m Specific Humidity |
| 5 | `rain` | Precipitation (log1p normalized) |
| 6 | `swdown` | Shortwave Downward Radiation |
| 7 | `t2` | 2m Temperature |
| 8 | **Δcpm25** | Temporal difference of PM2.5 (captures trends) |
| 9 | **ws10** | Wind speed = √(u10² + v10²) |
| 10 | **wsin** | Wind direction sine = sin(atan2(v10, u10)) |
| 11 | **wcos** | Wind direction cosine = cos(atan2(v10, u10)) |
| 12 | **pblh_inv** | Binary inversion flag (pblh < 300m → trapping conditions) |
| 13 | **vent** | Ventilation index = pblh × wind_speed (dispersion capacity) |

### Key Engineering Decisions

- **Dropped Features**: NH3, NMVOC_e, NOx, PM25_emission, SO2, bio — all zeros in the dataset
- **Log1p Transform**: Applied to `cpm25` and `rain` to handle skewed distributions
- **Z-Score Normalization**: Global mean/std computed across all 4 training months
- **Wind Decomposition**: Converted (u10, v10) → (speed, sin θ, cos θ) for rotationally meaningful representation

---

## 📊 Training Pipeline

### Data Splitting

```
Per-month split:
├── Training: timesteps [0 ... T-88]
├── Gap: 16 hours (prevents leakage)
└── Validation: last 72 hours
```

### Episode-Weighted Sampling

We detect **pollution episodes** (anomalous PM2.5 spikes) using a rolling-window z-score method:

- **Window**: 24 hours
- **Threshold**: 1.5σ above rolling mean
- **Weighting**: Above-median episode density → **4× weight**, top 10% → **8× weight**
- **Effect**: Model sees pollution spikes 4-8× more often during training, directly improving Episode SMAPE

### Loss Function — Asymmetric Combined Loss

```
L = 0.45 × Huber + 0.25 × wSMAPE + 0.15 × CorrLoss + 0.15 × AsymLoss
```

| Component | Purpose |
|---|---|
| **Huber Loss** (δ=1.0) | Stable gradients, robust to outliers |
| **Weighted SMAPE** | Magnitude-weighted symmetric percentage error — directly optimizes competition metric |
| **Correlation Loss** | 1 − Pearson(pred, true) per spatial frame — optimizes Episode Correlation |
| **Asymmetric Loss** | Extra penalty for underpredicting high-PM2.5 cells (target > 0.5σ) — targets Episode SMAPE |

### Optimizer & Scheduler

- **AdamW**: lr=5e-4, weight_decay=1e-4
- **Warmup**: 2-3 epoch linear warmup
- **Cosine Annealing**: Smooth decay after warmup
- **SWA (Stochastic Weight Averaging)**: Activated at epoch 10-18 with lr=1e-4 for flatter minima and better generalization
- **Gradient Clipping**: Max norm = 2.0

### Data Augmentation

- **Horizontal flip** (50% probability)
- **Vertical flip** (50% probability)
- Applied independently to both input and target

---

## 🎯 Inference — 4-Flip Test-Time Augmentation (TTA)

```
Final = mean(
    predict(original),
    predict(h_flip) → un_h_flip,
    predict(v_flip) → un_v_flip,
    predict(hv_flip) → un_hv_flip
)
```

4 forward passes averaged to reduce prediction variance by ~50%, providing a consistent +0.3-0.5% boost.

---

## 📈 Results

| Model | Val Composite | LB Score | Params |
|---|---|---|---|
| Lucifer (baseline) | 0.9189 | 0.8785 | 18.3M |
| **V15 (TCN+BiLSTM+GRU)** | — | **0.8815** | ~5M |
| V17 (STUNet speed) | — | — | ~4.5M |
| **Ensemble (V15+V17)** | — | **TBD** | ~9.5M total |

### Key Insight: Capacity vs. Generalization

Our best standalone model (Lucifer, 18.3M params) achieved 0.9189 on validation but only 0.8785 on the leaderboard — a **0.04 overfitting gap**. By reducing model capacity to ~5M params and increasing regularization, V15 achieved **0.8815 on LB** despite lower validation scores, demonstrating that **generalization trumps memorization** for this task.

---

## 🏗️ Technical Stack

| Component | Technology |
|---|---|
| **Framework** | PyTorch 2.x |
| **Hardware** | NVIDIA Tesla T4 (16GB VRAM) |
| **Precision** | Mixed (FP16/FP32 via torch.amp) |
| **Platform** | Kaggle Notebooks |
| **Training Time** | V15: ~58 min, V17: ~12 min |

---

## 📁 Repository Structure

```
AISE/
├── README.md                 # This file
├── lucifer.ipynb             # Baseline model (18.3M params, LB=0.8785)
├── phase2-v15.ipynb          # TCN+BiLSTM+GRU+Attention (LB=0.8815) ⭐
├── phase2-v17.ipynb          # STUNet speed run + ensemble with V15
├── phase2-v14.ipynb          # Reduced-capacity STUNet
├── phase2-v16.ipynb          # BiConvLSTM+ResUNet variant
├── phase2-v9.ipynb           # Earlier BiConvLSTM baseline (LB=0.8785)
└── phase2-v6.ipynb           # Foundation BiConvLSTM model
```

---

## 🔑 Key Innovations

1. **Multi-Architecture Ensemble**: Combining temporal-recurrent (TCN+BiLSTM+GRU) with purely convolutional (STUNet) models for maximum diversity

2. **Episode-Weighted Training**: Custom sampling strategy that oversamples pollution spike events by 4-8×, directly targeting the competition's episode metrics

3. **Asymmetric Loss Design**: Penalizing underprediction of high-PM2.5 events more heavily, aligned with the competition's emphasis on episode accuracy

4. **Capacity-Controlled Generalization**: Deliberately limiting model size (5M vs 18M params) to prevent overfitting to 2016 training data while maintaining accuracy on 2017 test data

5. **Physics-Informed Features**: Wind decomposition, ventilation index, and boundary layer inversion detection encode atmospheric dispersion physics directly into the input

6. **Stochastic Weight Averaging**: SWA during final training epochs finds flatter minima that generalize better across temporal distribution shifts (2016→2017)

---

## 👥 Team HackSprint

**Competition**: AISEHack Phase 2 — ANRF/IIT Delhi
**Theme**: Country-Level PM2.5 Concentration Forecasting
**Approach**: Ensemble Deep Learning with Physics-Informed Feature Engineering

---

*Built with 🔥 by Team HackSprint*
