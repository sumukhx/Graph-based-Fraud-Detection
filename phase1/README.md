
# Phase 1: Graph Heuristics Scoring Engine (Cold-Start)

This directory contains the Phase 1 implementation of the Graph-Based Fraud Detection Platform. It implements a **Time-Safe Scoring Engine** that iterates through transactions and computes risk scores based on graph heuristics, without using any machine learning models.

## 🚀 Quick Start

```bash
# Run the scoring engine (requires Phase 0 setup)
python phase1_heuristics.py
```

**Output**: `phase1_scores.csv` containing risk scores (0-1), top reasons, and detailed evidence for each transaction.

## 🧠 Heuristics Implemented

The engine implements 6 Key Heuristics (H1-H6) to detect fraud patterns:

| ID | Name | Logic Description | Weight |
|----|------|-------------------|--------|
| **H1** | **Device Burst** | High weighted count of recent transactions sharing the same Device. | 0.22 |
| **H2** | **Card/Region Collision** | One Issuer used in many Regions OR one Region used by many Issuers rapidly. | 0.15 |
| **H3** | **Email Domain Reuse** | Email Address linked to too many distinct Cards or Devices. | 0.15 |
| **H4** | **Identity Rarity** | Rare Identity Profile (Global Freq < 5) suddenly appearing across multiple Cards. | 0.18 |
| **H5** | **Ring / Density** | High overlap of shared entities (Device, Email, etc.) in the 2-hop neighborhood. | 0.20 |
| **H6** | **Velocity** | Rapid sequence of transactions (gaps < 5 mins) on the same Card/Device. | 0.10 |

## ⚙️ Configuration (Default Pars)

The script uses the following calibrated default parameters (`phase1_heuristics.py`):

### Time Windows
- **Short**: 1 Hour (H1, H6)
- **Medium**: 6 Hours (H2, H3, H4)
- **Long**: 24 Hours (H5)
- **Recency Decay ($\tau$)**: 2 Hours (Exponential decay weighting)

### Hub Control
- **Threshold**: Degree > 500
- **Strategy**: `neighbor_cap` (Limits traversal to most recent 50 neighbors for hubs to prevent explosion).

### Risk Aggregation
$$ Risk = 1 - \prod (1 - w_i \cdot s_i) $$
- **Low Risk**: < 0.30
- **Medium Risk**: 0.30 - 0.60
- **High Risk**: 0.60 - 0.80
- **Critical Risk**: > 0.80

## 📂 Project Structure

- `phase1_heuristics.py`: Core logic.
  - Builds Heterogeneous Graph from Phase 0.
  - Implements `FraudScorer` class with time-safe sliding windows.
  - Outputs CSV and prints evaluation metrics (PR-AUC).
- `phase1_scores.csv`: The result file.
  - Columns: `TransactionID`, `TransactionDT`, `risk_score`, `top_reasons` (explainable textual summary), `evidence_json`.
- `README.md`: This documentation.

## 📋 Requirements

- **Python 3.8+**
- `pandas`, `networkx`, `numpy`
- `scikit-learn` (Optional, for Evaluation metrics only)
