# Phase 2 V3 Metrics (Inductive Zero-Leakage)

## 1. Test Set Ranking (Target metrics)
* **PR-AUC**: 0.2575
* **ROC-AUC**: 0.7919
* **Recall @ 5% FPR**: 0.4479
* **Recall @ Top 1.0%**: 0.1354
* **Lift @ Top 1.0%**: 13.54X

## 2. Calibration (Validation Fit)
* **Brier Score (Raw)**: 0.0533
* **Brier Score (Calibrated)**: 0.0288

## 3. Confusion Matrix @ 5% FPR Validation Threshold (0.4415)
* TN: 2765 | FP: 139
* FN: 53 | TP: 43

## 4. Feature Importance Split (Gain)
* **Graph Embeddings**: 98.0%
* **Phase 1 Heuristics**: 0.6%
* **Raw Tabular**: 0.9%
* **Time-Safe Struct**: 0.4%

## Conclusions
The strict inductive separation prevents structural data leakage and provides our true capability baseline before implementing TGNs. 

## 5. Mandatory Ablation Study
| Feature Set | Test PR-AUC | Test Recall@5%FPR |
| ----------- | ----------- | ----------------- |
| Phase-1 Only | 0.0598 | 0.0938 |
| Embeddings Only | 0.1725 | 0.3646 |
| Raw Tabular Only | 0.1473 | 0.2812 |
| Embeddings + Phase-1 | 0.1765 | 0.3021 |
| All Features | 0.2575 | 0.4479 |
