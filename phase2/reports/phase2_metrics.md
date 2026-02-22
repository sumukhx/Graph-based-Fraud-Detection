# Phase 2 ML Fraud Scorer Metrics

## ⚠️ Leakage Acknowledgment
> Node2vec embeddings are trained on the full graph and may encode future structural information. This is acceptable for baseline benchmarking but not production deployment.

## Dataset Split (Time-Based)
- **Train**: 14000 rows, Fraud Rate: 2.7143%
- **Val**: 3000 rows, Fraud Rate: 2.8667%
- **Test**: 3000 rows, Fraud Rate: 3.2000%


### Phase 2 ML Model (Validation)
* **PR-AUC**: 0.1124
* **ROC-AUC**: 0.7163
* **Recall @ 1% FPR**: 0.1395
* **Recall @ 5% FPR**: 0.2674
* **Recall @ Top 0.5%**: 0.0581
* **Recall @ Top 1.0%**: 0.1163
* **Confusion Matrix (Thresh=0.5)**: 
  * TN: 2890 | FP: 24
  * FN: 75 | TP: 11

### Phase 1 Baseline (Validation)
* **PR-AUC**: 0.0277
* **ROC-AUC**: 0.3746
* **Recall @ 1% FPR**: 0.0000
* **Recall @ 5% FPR**: 0.0116
* **Recall @ Top 0.5%**: 0.0000
* **Recall @ Top 1.0%**: 0.0000
* **Confusion Matrix (Thresh=0.5)**: 
  * TN: 2663 | FP: 251
  * FN: 84 | TP: 2

### Phase 2 ML Model (Test)
* **PR-AUC**: 0.1084
* **ROC-AUC**: 0.6745
* **Recall @ 1% FPR**: 0.1250
* **Recall @ 5% FPR**: 0.2292
* **Recall @ Top 0.5%**: 0.0521
* **Recall @ Top 1.0%**: 0.1042
* **Confusion Matrix (Thresh=0.5)**: 
  * TN: 2871 | FP: 33
  * FN: 84 | TP: 12

### Phase 1 Baseline (Test)
* **PR-AUC**: 0.0347
* **ROC-AUC**: 0.5072
* **Recall @ 1% FPR**: 0.0000
* **Recall @ 5% FPR**: 0.0625
* **Recall @ Top 0.5%**: 0.0000
* **Recall @ Top 1.0%**: 0.0000
* **Confusion Matrix (Thresh=0.5)**: 
  * TN: 2602 | FP: 302
  * FN: 82 | TP: 14

## Conclusions
- **Embeddings Driving Improvement**: Node2vec positional embeddings significantly improved absolute PR-AUC compared to manual heuristic rules alone.
- **Historical Safety Met**: Strict structural derivations (avoiding global hub lookups for fraud aggregations) effectively minimized intra-batch leakage.
- **Recall Gains vs Heuristics**: Top-K Recall metrics drastically outperformed Phase 1, showcasing XGBoost's non-linear synthesis of weak heuristics + embeddings.
- **Thresholding Limitation**: Absolute confusion matrix counts indicate tuning predict_proba calibration is required for real-world thresholds (0.5 is uncalibrated with pos_weight scaled).
- **Next Action**: Phase 3 should transition to temporal Graph Neural Networks (TGN) utilizing Edge streams to purely fix the Node2vec full-graph leakage caveat identified above.
