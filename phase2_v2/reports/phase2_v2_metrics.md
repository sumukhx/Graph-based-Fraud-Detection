# Phase 2 V2 XGBoost Model Metrics

## Calibration
Isotonic Regression applied strictly on the Validation set.
* **Test Brier Score (Raw)**: 0.1190
* **Test Brier Score (Calibrated)**: 0.0313

## Dataset Split (Time-Based)
- **Train**: 14000 rows, Fraud Rate: 2.71%
- **Val**: 3000 rows, Fraud Rate: 2.87%
- **Test**: 3000 rows, Fraud Rate: 3.20%

### Test Set Performance (Calibrated)
* **PR-AUC**: 0.0785 (Target $\ge$ 0.18)
* **ROC-AUC**: 0.6855 (Target $\ge$ 0.75)

#### Precision/Recall Constraints
* **Recall @ 1.0% FPR**: 0.0417
* **Recall @ 5.0% FPR**: 0.1771 (Target $\ge$ 30%)

#### Value Generation (Lifts)
* **Lift @ Top 1.0%**: 4.17X
* **Lift @ Top 2.0%**: 5.21X
* **Lift @ Top 5.0%**: 3.96X

#### Confusion Matrix (Optimized Threshold = 0.2000)
* TN: 2815 | FP: 89
* FN: 82 | TP: 14

## Conclusions
- Deep feature interaction and calibration significantly adjust absolute capabilities.
- Overlap of train/val PR-AUC indicates generalization boundaries.
- **Next Step:** Compare directly against Phase 1 and Phase 2 V1 models.
