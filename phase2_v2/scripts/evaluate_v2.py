import os
import sys
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.calibration import IsotonicRegression, calibration_curve
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc, roc_curve, confusion_matrix, recall_score, brier_score_loss
import shap

# --- CONFIGURATION ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE2_V2_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

LOG_FILE = os.path.join(PHASE2_V2_DIR, "logs", "evaluate_v2.log")
PROCESSED_DATA_INPUT = os.path.join(PHASE2_V2_DIR, "artifacts", "processed_data.parquet")
MODEL_INPUT = os.path.join(PHASE2_V2_DIR, "artifacts", "xgboost_v2.model")
REPORT_OUTPUT = os.path.join(PHASE2_V2_DIR, "reports", "phase2_v2_metrics.md")

os.makedirs(os.path.join(PHASE2_V2_DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(PHASE2_V2_DIR, "reports"), exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[
    logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)
])

def compute_lift(y_true, y_prob, percentiles=[0.01, 0.02, 0.05]):
    df = pd.DataFrame({'true': y_true, 'prob': y_prob})
    df = df.sort_values('prob', ascending=False)
    
    total_fraud = y_true.sum()
    total_count = len(y_true)
    base_rate = total_fraud / total_count if total_count > 0 else 0
    
    lifts = {}
    for p in percentiles:
        n_top = int(total_count * p)
        if n_top == 0: continue
        top_df = df.head(n_top)
        top_fraud = top_df['true'].sum()
        top_rate = top_fraud / n_top
        lift = top_rate / base_rate if base_rate > 0 else 0
        lifts[f"Lift @ Top {p*100:.1f}%"] = lift
    return lifts
    
def get_recall_at_fpr(y_true, y_prob, target_fprs=[0.01, 0.05]):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    recalls = {}
    for t_fpr in target_fprs:
        # find index where fpr is closest to but <= target_fpr
        valid_idx = np.where(fpr <= t_fpr)[0]
        if len(valid_idx) > 0:
            best_idx = valid_idx[-1]
            recalls[f"Recall @ {t_fpr*100:.1f}% FPR"] = tpr[best_idx]
        else:
            recalls[f"Recall @ {t_fpr*100:.1f}% FPR"] = 0.0
    return recalls

def find_optimal_threshold(y_true, y_prob):
    # Optimize for Max F1
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * recall * precision / (recall + precision + 1e-9)
    # The last threshold is sometimes omitted so arrays match
    if len(f1_scores) == len(thresholds) + 1:
        f1_scores = f1_scores[:-1]
    
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    return best_thresh, best_f1
    
def run():
    logging.info("=== Starting Phase 2 V2 Calibration & Evaluation ===")
    
    logging.info(f"Loading data from {PROCESSED_DATA_INPUT}")
    df = pd.read_parquet(PROCESSED_DATA_INPUT)
    
    logging.info(f"Loading XGBoost model from {MODEL_INPUT}")
    model = xgb.XGBClassifier()
    model.load_model(MODEL_INPUT)
    
    # 1. Recover Split
    df = df.sort_values('TransactionDT').reset_index(drop=True)
    train_end = 14000
    val_end = 14000 + 3000
    test_end = 14000 + 3000 + 3000
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:test_end]
    
    features = [c for c in df.columns if c not in ['TransactionID', 'TransactionDT', 'isFraud']]
    
    X_train, y_train = train_df[features], train_df['isFraud'].astype(int)
    X_val, y_val = val_df[features], val_df['isFraud'].astype(int)
    X_test, y_test = test_df[features], test_df['isFraud'].astype(int)
    
    # 2. Raw Predictions
    logging.info("Getting raw model predictions...")
    raw_prob_val = model.predict_proba(X_val)[:, 1]
    raw_prob_test = model.predict_proba(X_test)[:, 1]
    raw_prob_train = model.predict_proba(X_train)[:, 1]
    
    # Check Overfitting
    prec_train, rec_train, _ = precision_recall_curve(y_train, raw_prob_train)
    prauc_train = auc(rec_train, prec_train)
    
    prec_val, rec_val, _ = precision_recall_curve(y_val, raw_prob_val)
    prauc_val = auc(rec_val, prec_val)
    
    logging.info(f"Train PR-AUC: {prauc_train:.4f} | Val PR-AUC: {prauc_val:.4f}")
    if prauc_train - prauc_val > 0.2:
        logging.warning("High overfitting detected in raw model!")
    
    # 3. Calibration (Isotonic Regression on Val)
    logging.info("Fitting Isotonic Regression on Validation Set...")
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(raw_prob_val, y_val)
    
    cal_prob_val = ir.predict(raw_prob_val)
    cal_prob_test = ir.predict(raw_prob_test)
    
    brier_raw = brier_score_loss(y_test, raw_prob_test)
    brier_cal = brier_score_loss(y_test, cal_prob_test)
    logging.info(f"Brier Score (Raw Test): {brier_raw:.4f}")
    logging.info(f"Brier Score (Calibrated Test): {brier_cal:.4f}")
    
    # 4. Threshold Optimization (on Val)
    best_thresh, best_f1 = find_optimal_threshold(y_val, cal_prob_val)
    logging.info(f"Optimal Threshold (Max F1 on Val): {best_thresh:.4f}")
    
    # 5. Deep Evaluation (on Test)
    logging.info("Calculating final metrics on Test Set...")
    y_test_pred = (cal_prob_test >= best_thresh).astype(int)
    
    roc_auc = roc_auc_score(y_test, cal_prob_test)
    prec_t, rec_t, _ = precision_recall_curve(y_test, cal_prob_test)
    pr_auc = auc(rec_t, prec_t)
    
    lifts = compute_lift(y_test, cal_prob_test)
    recalls = get_recall_at_fpr(y_test, cal_prob_test)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    
    # Prepare Report
    report = f"""# Phase 2 V2 XGBoost Model Metrics

## Calibration
Isotonic Regression applied strictly on the Validation set.
* **Test Brier Score (Raw)**: {brier_raw:.4f}
* **Test Brier Score (Calibrated)**: {brier_cal:.4f}

## Dataset Split (Time-Based)
- **Train**: {len(y_train)} rows, Fraud Rate: {y_train.mean()*100:.2f}%
- **Val**: {len(y_val)} rows, Fraud Rate: {y_val.mean()*100:.2f}%
- **Test**: {len(y_test)} rows, Fraud Rate: {y_test.mean()*100:.2f}%

### Test Set Performance (Calibrated)
* **PR-AUC**: {pr_auc:.4f} (Target $\ge$ 0.18)
* **ROC-AUC**: {roc_auc:.4f} (Target $\ge$ 0.75)

#### Precision/Recall Constraints
* **{list(recalls.keys())[0]}**: {list(recalls.values())[0]:.4f}
* **{list(recalls.keys())[1]}**: {list(recalls.values())[1]:.4f} (Target $\ge$ 30%)

#### Value Generation (Lifts)
* **{list(lifts.keys())[0]}**: {list(lifts.values())[0]:.2f}X
* **{list(lifts.keys())[1]}**: {list(lifts.values())[1]:.2f}X
* **{list(lifts.keys())[2]}**: {list(lifts.values())[2]:.2f}X

#### Confusion Matrix (Optimized Threshold = {best_thresh:.4f})
* TN: {tn} | FP: {fp}
* FN: {fn} | TP: {tp}

## Conclusions
- Deep feature interaction and calibration significantly adjust absolute capabilities.
- Overlap of train/val PR-AUC indicates generalization boundaries.
- **Next Step:** Compare directly against Phase 1 and Phase 2 V1 models.
"""
    
    with open(REPORT_OUTPUT, "w") as f:
        f.write(report)
        
    logging.info(f"Evaluation report generated at: {REPORT_OUTPUT}")
    
    # 6. SHAP Execution
    logging.info("Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    # Background dataset for speed
    bg = X_test.sample(500, random_state=42) if len(X_test) > 500 else X_test
    shap_values = explainer.shap_values(bg)
    
    # Get top 20 features
    vals = np.abs(shap_values).mean(0)
    feature_importance = pd.DataFrame(list(zip(features, vals)), columns=['col_name', 'feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
    
    logging.info("\n--- Top 20 SHAP Features ---")
    for idx, row in feature_importance.head(20).iterrows():
        logging.info(f"{row['col_name']}: {row['feature_importance_vals']:.4f}")
        
    logging.info("=== Calibration & Evaluation Complete ===")

if __name__ == "__main__":
    run()
