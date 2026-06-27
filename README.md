# Intelligent Intrusion Detection System for Encrypted Network Traffic

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-orange)](https://xgboost.readthedocs.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)](https://pytorch.org)
[![SHAP](https://img.shields.io/badge/SHAP-0.51-purple)](https://shap.readthedocs.io)
[![Dataset](https://img.shields.io/badge/Dataset-CICIDS2017-green)](https://www.unb.ca/cic/datasets/ids-2017.html)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

> A production-grade, hybrid Intrusion Detection System that combines a high-accuracy supervised machine learning classifier (XGBoost) with an unsupervised deep learning Autoencoder (PyTorch) to detect **both known attacks and zero-day anomalies** in network traffic — with full SHAP-based explainability for SOC analysts.

---

## Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Key Results](#key-results)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running the Pipeline](#running-the-pipeline)
- [Phase-by-Phase Breakdown](#phase-by-phase-breakdown)
- [Explainability (SHAP)](#explainability-shap)
- [Generalization Testing](#generalization-testing)
- [Limitations & Future Work](#limitations--future-work)
- [Technologies Used](#technologies-used)

---

## Project Overview

Traditional IDS solutions rely entirely on known attack signatures, leaving networks vulnerable to novel (zero-day) exploits. This project presents a **Hybrid IDS** that solves this gap using a two-stage decision engine:

1. **Stage 1 — Known Attack Detection (XGBoost):** A gradient-boosted tree classifier trained on 2.2 million labelled network flows from the CICIDS2017 dataset. Achieves 99.89% accuracy on known attack families including DoS, DDoS, PortScan, and Botnet traffic.

2. **Stage 2 — Zero-Day Anomaly Detection (Autoencoder):** A PyTorch Encoder-Decoder neural network trained *exclusively* on benign traffic. Any packet that XGBoost passes as "safe" is then reconstructed by the Autoencoder. A high reconstruction error (MSE > 0.0116) flags the packet as an unknown anomaly.

The system is fully **explainable** via SHAP (SHapley Additive exPlanations), enabling SOC analysts to understand exactly *why* any packet was flagged as malicious.

---

## System Architecture

```
 Incoming Network Traffic (Flow Features)
               │
               ▼
    ┌─────────────────────┐
    │   Preprocessing     │  ← Clean, normalise, encode labels
    └─────────┬───────────┘
              │
              ▼
    ┌─────────────────────┐
    │  XGBoost Classifier │  ← Stage 1: Known Attack Detection
    └─────────┬───────────┘
              │
    ┌─────────┴───────────┐
    │                     │
  Attack ✓          Benign? (uncertain)
    │                     │
    │          ┌──────────▼──────────┐
    │          │  PyTorch Autoencoder│  ← Stage 2: Zero-Day Detection
    │          │  (Reconstruction    │
    │          │   Error Analysis)   │
    │          └──────────┬──────────┘
    │                     │
    │          ┌──────────┴──────────┐
    │          │                     │
    │      MSE > 0.0116         MSE ≤ 0.0116
    │      (Anomaly) ✓           (Safe) ✓
    │          │
    └──────────▼
         ALERT / BLOCK
```

---

## Key Results

### Performance on CICIDS2017 Test Set (442,406 flows)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|
| Random Forest (Baseline) | 99.85% | 99.75% | 99.62% | 99.69% | 99.99% |
| XGBoost (Optimised) | 99.89% | 99.74% | 99.79% | 99.77% | 99.99% |
| Autoencoder (Standalone) | 83.54% | 92.42% | 35.57% | 51.37% | 79.87% |
| **Hybrid IDS (Final System)** | **99.17%** | **96.92%** | **99.79%** | **98.33%** | **99.99%** |

> **Note:** The Autoencoder's lower standalone recall is expected — it was designed to detect *unknown* anomalies, not classify known labelled attack families. Its real strength emerges inside the Hybrid system.

### Attack Families Detected (CICIDS2017)

| Attack Type | Samples |
|:---|---:|
| DoS Hulk | 230,124 |
| PortScan | 158,804 |
| DDoS | 128,027 |
| DoS GoldenEye | 10,293 |
| DoS Slowloris | 5,796 |
| DoS Slowhttptest | 5,499 |
| Bot | 1,956 |
| Infiltration | 36 |
| Heartbleed | 11 |

### Feature Optimisation

Our Phase 4 feature importance analysis reduced the feature space from **78 → 39 features** while retaining **99% of the model's predictive power**, halving inference time with zero accuracy loss.

---

## Dataset

| Dataset | Purpose | Rows | Features |
|:---|:---|---:|---:|
| **CICIDS2017** (`combine.csv`) | Training & evaluation | 2,214,469 | 78 |
| **UNSW-NB15** | Generalization testing | 246,996 | 45 |

- **CICIDS2017:** [University of New Brunswick](https://www.unb.ca/cic/datasets/ids-2017.html) — Place as `data/raw/combine.csv`
- **UNSW-NB15:** [Kaggle — mrwellsdavid/unsw-nb15](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15) — Place CSV files in `data/raw/`

---

## Project Structure

```
Intelligent-Intrusion-Detection-System-for-Encrypted-Network-Traffic/
│
├── data/
│   ├── raw/
│   │   ├── combine.csv                    ← CICIDS2017 raw dataset
│   │   └── UNSW_NB15_testing-set.csv      ← UNSW-NB15 generalization dataset
│   └── processed/
│       ├── cleaned_data.csv               ← After Phase 1 preprocessing
│       └── optimized_data.csv             ← After Phase 4 feature selection
│
├── models/
│   ├── random_forest_baseline.pkl         ← Trained Random Forest
│   ├── xgboost_baseline.pkl               ← Trained XGBoost (primary classifier)
│   ├── autoencoder.pth                    ← PyTorch Autoencoder weights
│   └── ae_scaler.pkl                      ← MinMaxScaler for Autoencoder input
│
├── results/
│   ├── baseline_metrics.json              ← Phase 3 evaluation metrics
│   ├── ae_threshold.json                  ← Autoencoder anomaly threshold
│   ├── evaluation_metrics.json            ← Phase 7 full comparison metrics
│   ├── generalization_metrics.json        ← Phase 8 UNSW-NB15 results
│   ├── class_distribution.png             ← Phase 2 EDA
│   ├── correlation_heatmap.png            ← Phase 2 EDA
│   ├── top_features_histograms.png        ← Phase 2 EDA
│   ├── feature_importance.png             ← Phase 4 XGBoost importances
│   ├── cumulative_importance.png          ← Phase 4 feature selection curve
│   ├── reconstruction_error.png           ← Phase 5 Autoencoder MSE dist.
│   ├── hybrid_confusion_matrix.png        ← Phase 6 Hybrid IDS matrix
│   ├── model_comparison.png               ← Phase 7 ROC-AUC + metrics bar
│   ├── generalization_confusion_matrix.png← Phase 8 cross-dataset matrix
│   ├── shap_summary_bar.png               ← Phase 9 global importance
│   ├── shap_beeswarm.png                  ← Phase 9 feature direction
│   └── shap_waterfall.png                 ← Phase 9 single prediction
│
├── scripts/
│   ├── preprocess.py                      ← Phase 1: Data cleaning & encoding
│   ├── eda.py                             ← Phase 2: Exploratory analysis
│   ├── train_baseline.py                  ← Phase 3: RF + XGBoost training
│   ├── feature_optimization.py            ← Phase 4: Feature selection
│   ├── train_autoencoder.py               ← Phase 5: PyTorch Autoencoder
│   ├── hybrid_ids.py                      ← Phase 6: Hybrid decision engine
│   ├── evaluate.py                        ← Phase 7: Full model comparison
│   ├── generalization_test.py             ← Phase 8: Cross-dataset testing
│   └── shap_explain.py                    ← Phase 9: SHAP explainability
│
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### Prerequisites
- Python 3.10 or higher
- pip

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Intelligent-Intrusion-Detection-System-for-Encrypted-Network-Traffic.git
cd Intelligent-Intrusion-Detection-System-for-Encrypted-Network-Traffic
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
pip install torch
```

> **Note:** TensorFlow 2.16.1 (listed in requirements.txt) is not compatible with Python 3.14+. This project uses PyTorch as an equivalent alternative for the Autoencoder.

### 3. Download Datasets
- **CICIDS2017** → Place as `data/raw/combine.csv`
- **UNSW-NB15** (optional, for Phase 8) → Place CSV files in `data/raw/`

---

## Running the Pipeline

Run each phase sequentially from the project root directory:

```bash
# Phase 1 — Preprocess the raw dataset
python scripts/preprocess.py

# Phase 2 — Exploratory Data Analysis
python scripts/eda.py

# Phase 3 — Train baseline ML models (Random Forest + XGBoost)
python scripts/train_baseline.py

# Phase 4 — Feature optimisation (select top 39 features)
python scripts/feature_optimization.py

# Phase 5 — Train the Autoencoder on normal traffic only
python scripts/train_autoencoder.py

# Phase 6 — Run the Hybrid IDS decision engine
python scripts/hybrid_ids.py

# Phase 7 — Formal evaluation: compare all models with ROC-AUC curves
python scripts/evaluate.py

# Phase 8 — Generalization test on UNSW-NB15 (requires dataset in data/raw/)
python scripts/generalization_test.py

# Phase 9 — SHAP explainability plots
python scripts/shap_explain.py
```

All output plots are saved to `results/` and models are saved to `models/`.

---

## Phase-by-Phase Breakdown

| Phase | Script | Description | Output |
|:---:|:---|:---|:---|
| 0 | — | Environment setup & project structure | `data/`, `models/`, `results/`, `scripts/` |
| 1 | `preprocess.py` | Clean data, drop NaN/Inf, binary-encode labels | `cleaned_data.csv` |
| 2 | `eda.py` | Class distribution, correlation heatmap, histograms | 3 PNG plots |
| 3 | `train_baseline.py` | Train Random Forest (50 trees) + XGBoost (100 estimators) | 2 `.pkl` model files |
| 4 | `feature_optimization.py` | Extract importances, reduce 78 → 39 features (99% power retained) | `optimized_data.csv`, 2 plots |
| 5 | `train_autoencoder.py` | Train PyTorch Encoder-Decoder on benign traffic only; calculate MSE threshold | `autoencoder.pth`, `ae_threshold.json` |
| 6 | `hybrid_ids.py` | Two-stage decision: XGBoost → Autoencoder for benign packets | Confusion matrix plot |
| 7 | `evaluate.py` | Head-to-head comparison + ROC-AUC curves for all 4 models | `evaluation_metrics.json`, comparison plot |
| 8 | `generalization_test.py` | Cross-dataset test on UNSW-NB15 with semantic feature mapping | `generalization_metrics.json`, matrix plot |
| 9 | `shap_explain.py` | SHAP TreeExplainer: Summary Bar, Beeswarm, Waterfall plots | 3 SHAP PNG plots |

---

## Explainability (SHAP)

Model explainability is critical for real-world SOC deployment. Using SHAP (SHapley Additive exPlanations) with `TreeExplainer` on the XGBoost model, the system can:

- **Globally** rank which network features most commonly drive attack predictions.
- **Locally** explain *why* a specific packet was flagged — step-by-step, feature by feature.

Top contributing features identified:
1. `Average Packet Size`
2. `Bwd Packet Length Std`
3. `Bwd Header Length`
4. `Max Packet Length`
5. `Fwd IAT Max`
6. `Destination Port`
7. `Init_Win_bytes_forward`
8. `Flow IAT Mean`

---

## Generalization Testing

Testing the Hybrid IDS on the **UNSW-NB15** dataset revealed significant performance degradation, which is an important and honest finding:

| Dataset | Accuracy | F1-Score |
|:---|:---:|:---:|
| CICIDS2017 (trained on) | 99.17% | 98.33% |
| UNSW-NB15 (unseen) | 42.95% | 0.06% |

**Root Cause — Dataset Shift:**
- Only ~24 of 78 required features could be semantically mapped from UNSW-NB15, with remaining features zeroed out.
- The attack families (Fuzzer, Exploit, Reconnaissance, Backdoor) are statistically distinct from CICIDS2017 attack families.
- Both datasets were generated in different lab environments.

**Implications for Future Work:**
This result motivates several important research directions including transfer learning, domain adaptation, and online/continual learning for real-world IDS deployments.

---

## Limitations & Future Work

| Limitation | Proposed Future Work |
|:---|:---|
| Model trained on static lab data | Implement **online learning** for continuous adaptation |
| Low cross-dataset generalization | Explore **transfer learning** and **domain adaptation** |
| Autoencoder threshold is fixed | Use **reinforcement learning** for dynamic threshold adjustment |
| Feature-based (requires CICFlowMeter) | Integrate with live packet capture for **real-time inference** |
| No multi-class classification | Extend to **multi-class IDS** to identify specific attack families |

---

## Technologies Used

| Library | Version | Purpose |
|:---|:---:|:---|
| `pandas` | 2.2.1 | Data manipulation |
| `numpy` | 1.26.4 | Numerical computing |
| `scikit-learn` | 1.4.1 | ML models, preprocessing, metrics |
| `xgboost` | 2.0.3 | Gradient-boosted classifier |
| `torch` (PyTorch) | 2.12.1 | Autoencoder deep learning |
| `shap` | 0.51.0 | Model explainability |
| `matplotlib` | 3.8.3 | Visualizations |
| `seaborn` | 0.13.2 | Statistical plots |
| `joblib` | 1.3.2 | Model serialization |
| `imbalanced-learn` | 0.12.0 | Class imbalance handling |

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <sub>Built as part of a Security Operations Centre (SOC) Research Project.</sub>
</div>
=======
# Intelligent-Intrusion-Detection-System-for-Encrypted-Network-Traffic
