import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix, brier_score_loss
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt
import json
import logging
import os
from glob import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = "test_audit/artifacts"
REPORTS_DIR = "test_audit/reports"

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_threshold_for_recall(y_true, y_pred, target_fpr):
    thresholds = sorted(list(set(y_pred)), reverse=True)
    best_thresh = thresholds[0]
    for t in thresholds:
        preds = (y_pred >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        if fpr >= target_fpr:
            break
        best_thresh = t
    return best_thresh

def evaluate_version(df_val, df_test, v_name):
    y_val = df_val['y_true'].values
    preds_val = df_val['y_score'].values
    y_test = df_test['y_true'].values
    preds_test = df_test['y_score'].values
    
    # Ranking
    pr_auc = average_precision_score(y_test, preds_test)
    roc_auc = roc_auc_score(y_test, preds_test)
    
    # Brier
    brier_raw = brier_score_loss(y_test, preds_test)
    
    # Calibrate (Valid-Fit constraint)
    iso = IsotonicRegression(out_of_bounds='clip')
    try:
        iso.fit(preds_val, y_val)
        preds_test_calib = iso.predict(preds_test)
        brier_calib = brier_score_loss(y_test, preds_test_calib)
    except:
        brier_calib = brier_raw
    
    # Budget Constraints
    def get_recall_at_fpr(y, p, thresh):
        preds = (p >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
        return tp / (tp + fn) if (tp + fn) > 0 else 0
        
    thresh_1fpr = get_threshold_for_recall(y_val, preds_val, 0.01)
    thresh_5fpr = get_threshold_for_recall(y_val, preds_val, 0.05)
    
    rec_1fpr = get_recall_at_fpr(y_test, preds_test, thresh_1fpr)
    rec_5fpr = get_recall_at_fpr(y_test, preds_test, thresh_5fpr)
    
    # Top-K Budget
    def top_k_metrics(y, p, pct):
        k = max(1, int(len(y) * pct))
        idx = np.argsort(p)[-k:]
        rec = sum(y[idx]) / sum(y)
        base = sum(y) / len(y)
        lift = (sum(y[idx]) / k) / base if base > 0 else 0
        return rec, lift
        
    rec_top05, _ = top_k_metrics(y_test, preds_test, 0.005)
    rec_top1, lift_top1 = top_k_metrics(y_test, preds_test, 0.01)
    _, lift_top2 = top_k_metrics(y_test, preds_test, 0.02)
    _, lift_top5 = top_k_metrics(y_test, preds_test, 0.05)
    
    # Optimal CM (at Val 5% FPR)
    preds_bin = (preds_test >= thresh_5fpr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds_bin).ravel()
    
    return {
        "Version": v_name,
        "PR_AUC": pr_auc,
        "ROC_AUC": roc_auc,
        "Brier_Raw": brier_raw,
        "Brier_Calib": brier_calib,
        "Rec_1_FPR": rec_1fpr,
        "Rec_5_FPR": rec_5fpr,
        "Rec_Top05": rec_top05,
        "Rec_Top1": rec_top1,
        "Lift_Top1": lift_top1,
        "Lift_Top2": lift_top2,
        "Lift_Top5": lift_top5,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp
    }

def generate_comparisons():
    ensure_dir(f"{REPORTS_DIR}/figures")
    
    with open(f"{ARTIFACTS_DIR}/split_index.json", "r") as f:
        splits = json.load(f)
        
    val_idx = set(splits['val'])
    test_idx = set(splits['test'])
    
    pred_files = glob(f"{ARTIFACTS_DIR}/version_preds/*.parquet")
    logger.info(f"Evaluating {len(pred_files)} saved versions...")
    
    results = []
    
    for pf in pred_files:
        v_name = os.path.basename(pf).replace(".parquet", "")
        df = pd.read_parquet(pf)
        
        df_val = df[df['TransactionID'].isin(val_idx)]
        df_test = df[df['TransactionID'].isin(test_idx)]
        
        res = evaluate_version(df_val, df_test, v_name)
        results.append(res)
        
    df_res = pd.DataFrame(results).sort_values("PR_AUC", ascending=False)
    df_res.to_csv(f"{REPORTS_DIR}/benchmark_summary.csv", index=False)
    logger.info("Evaluation complete. Metrics saved to reports/benchmark_summary.csv")

if __name__ == "__main__":
    generate_comparisons()
