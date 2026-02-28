import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score, confusion_matrix
import json
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = "phase2_v3/artifacts"
REPORTS_DIR = "phase2_v3/reports"

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

def train_and_eval(df_train, df_val, df_test, feature_group, name):
    X_train, y_train = df_train[feature_group], df_train['isFraud']
    X_val, y_val = df_val[feature_group], df_val['isFraud']
    X_test, y_test = df_test[feature_group], df_test['isFraud']
    
    num_pos = y_train.sum()
    scale_weight = (len(y_train) - num_pos) / num_pos if num_pos > 0 else 1.0

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
        'max_depth': 5,
        'min_child_weight': 10,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'learning_rate': 0.05,
        'reg_lambda': 10,
        'reg_alpha': 1,
        'scale_pos_weight': scale_weight,
        'random_state': 42,
        'n_jobs': 4
    }
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=800,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=False
    )
    
    preds_val = model.predict(dval)
    preds_test = model.predict(dtest)
    
    test_pr_auc = average_precision_score(y_test, preds_test)
    thresh = get_threshold_for_recall(y_val, preds_val, 0.05)
    
    test_binary = (preds_test >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, test_binary).ravel()
    recall_5fpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    logger.info(f"[{name}] Test PR-AUC: {test_pr_auc:.4f} | Recall@5%FPR: {recall_5fpr:.4f}")
    return test_pr_auc, recall_5fpr

def run_ablation():
    logger.info("Loading feature matrix...")
    df = pd.read_parquet(f"{ARTIFACTS_DIR}/tx_feature_matrix.parquet")
    
    with open(f"{ARTIFACTS_DIR}/split_index.json", "r") as f:
        splits = json.load(f)
        
    df_train = df[df['TransactionID'].isin(set(splits['train']))]
    df_val = df[df['TransactionID'].isin(set(splits['val']))]
    df_test = df[df['TransactionID'].isin(set(splits['test']))]
    
    all_features = [c for c in df.columns if c not in ['TransactionID', 'TransactionDT', 'isFraud']]
    
    phase1_features = [c for c in all_features if c.startswith('phase1_')]
    emb_features = [c for c in all_features if c.startswith('emb_') or c.startswith('entity_seen_')]
    raw_features = [c for c in all_features if c.startswith('raw_')]
    struct_features = [c for c in all_features if c.startswith('struct_')]
    
    results = {}
    
    # 1. Phase-1 only
    results["Phase-1 Only"] = train_and_eval(df_train, df_val, df_test, phase1_features, "Phase-1 Only")
    
    # 2. Embeddings only
    results["Embeddings Only"] = train_and_eval(df_train, df_val, df_test, emb_features, "Embeddings Only")
    
    # 3. Raw tabular only
    results["Raw Tabular Only"] = train_and_eval(df_train, df_val, df_test, raw_features, "Raw Tabular Only")
    
    # 4. Embeddings + Phase-1
    results["Embeddings + Phase-1"] = train_and_eval(df_train, df_val, df_test, emb_features + phase1_features, "Embeddings + Phase-1")
    
    # 5. All features
    results["All Features"] = train_and_eval(df_train, df_val, df_test, all_features, "All Features")
    
    # Append to report
    ablation_md = "\n## 5. Mandatory Ablation Study\n"
    ablation_md += "| Feature Set | Test PR-AUC | Test Recall@5%FPR |\n"
    ablation_md += "| ----------- | ----------- | ----------------- |\n"
    for name, (pr, rec) in results.items():
        ablation_md += f"| {name} | {pr:.4f} | {rec:.4f} |\n"
        
    with open(f"{REPORTS_DIR}/phase2_v3_metrics.md", "a") as f:
        f.write(ablation_md)
    logger.info("Ablation study written to report!")

if __name__ == "__main__":
    run_ablation()
