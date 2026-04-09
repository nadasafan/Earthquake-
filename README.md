# SEEPS: Self-Evolving Earthquake Perception System

**Integrating Reinforcement Learning and Causal Inference for Short-Term Earthquake Early Warning**

## Overview

SEEPS is a unified framework for short-term earthquake prediction that couples three complementary learning paradigms:

1. **Residual Dilated Temporal Convolutional Network (TCN)** with Squeeze-and-Excitation attention for multi-scale waveform encoding
2. **Structural Causal Module** for counterfactual interpretation of precursor signals
3. **Reinforcement Learning Agent** (Dueling-DDQN) regularized by Elastic Weight Consolidation (EWC) for continual, deployment-time adaptation

## Architecture

```
Raw Waveform → Preprocessing → [Transformer Encoder + Statistical Features]
    → Hybrid Fusion → Residual Dilated TCN → SE Attention
    → [Optional StationGNN] → Multi-Horizon Heads (2/5/9 min)
    → Bayesian Decision Layer → Graded Alert
```

## Key Results

| Metric | Value |
|--------|-------|
| Cross-Validated ROC-AUC | 0.997 |
| Held-Out Class-1 F1 | 0.714 |
| Streaming F1 (~16k windows) | 0.907 |
| Mean Lead Time | 9.3 minutes |
| False-Alarm Reduction vs STA/LTA | 50.3% |

## Repository Structure

```
├── SEEPS_Complete.ipynb      # Full implementation (all components)
├── Seeqs Optimized.ipynb     # Experimental development notebook
├── SEEPS.docx                # Research paper
├── README.md
└── .gitignore
```

## Components Implemented

### Model Architecture
- **WaveformTransformerEncoder** — Self-attention encoder for raw waveform embedding (Section 3.C.1)
- **StatisticalFeatureExtractor** — RMS, ZCR, Hjorth parameters, spectral features (Section 3.C.2)
- **HybridFusionEncoder** — Learned fusion of deep + statistical pathways (Section 3.C.3)
- **ResidualDilatedBlock** — Dilated causal convolutions with residual connections (Section 3.D.1-2)
- **SqueezeExcitation** — Channel attention with global pooling and bottleneck MLP (Section 3.D.4)
- **TemporalConvNet** — Three-block TCN with dilations [1, 2, 4] (Section 3.D)
- **StationGNN** — Graph neural network for inter-station spatial dependencies
- **MultiHorizonHead** — Three sigmoid heads for 2/5/9-minute prediction horizons (Section 3.F.4)
- **StructuralCausalModule** — Counterfactual interpretability via do-calculus (Figure 9)
- **BayesianDecisionLayer** — Temperature-scaled confidence gating (Section 3.E.5)

### Reinforcement Learning
- **DuelingDDQN** — Dueling architecture with value/advantage streams (Section 3.E.3)
- **EWAlertReward** — Composite reward with lead-time bonus, miss penalty, and false-alarm cost (Section 3.E.2)
- **EWC** — Elastic Weight Consolidation for continual learning (Section 3.E.4)
- **PrioritizedReplayBuffer** — Experience replay weighted toward rare high-reward transitions (Section 3.E.6)

### Preprocessing
- **SeismicPreprocessor** — De-meaning, variance normalization, temporal standardization, Butterworth bandpass (Section 3.A)
- **EventCueing** — Cascaded STA/LTA + RMS-quantile gating (Section 3.B)

### Evaluation
- 5-Fold stratified cross-validation with 5-run averaging
- Ablation study (Table 6)
- Baseline comparisons: STA/LTA, Logistic Regression, Random Forest (Table 5)
- Large-scale streaming evaluation (~16,000 windows)
- Bootstrap 95% confidence intervals
- Paired permutation test
- Expected Calibration Error (ECE)

## Data

Seismic waveform data from the **Almaty regional seismic network (2023–2024)**, recorded at 100 Hz sampling rate on HHZ channels in Mini-SEED format. The dataset comprises 935 manually verified analysis windows with a ~9:1 class imbalance (no-quake vs. pre-quake).

## Requirements

```
torch >= 2.0
torch-geometric
obspy
scikit-learn
numpy
pandas
matplotlib
seaborn
scipy
tqdm
```

## Usage

1. Open `SEEPS_Complete.ipynb` in Google Colab
2. Configure data paths in the configuration cell
3. Run cells sequentially

## Citation

```
SEEPS: A Self-Evolving Earthquake Perception System Integrating
Reinforcement Learning and Causal Inference
```

## License

This project is for academic research purposes.
