import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score
import json
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = "test_audit/artifacts"

def run_label_shuffle_test():
    """
    Shuffles y_true in train, fits a quick mock tree. 
    If test PR-AUC > Fraud Rate by a margin, features MUST be leaking label distribution somehow.
    """
    logger.info("Running Label Shuffle Leakage Test on V3 features...")
    df = pd.read_parquet("phase2_v3/artifacts/tx_feature_matrix.parquet")
    with open(f"{ARTIFACTS_DIR}/split_index.json", "r") as f:
        splits = json.load(f)
        
    train_idx = set(splits['train'])
    test_idx = set(splits['test'])
    
    df_train = df[df['TransactionID'].isin(train_idx)].copy()
    df_test = df[df['TransactionID'].isin(test_idx)].copy()
    
    features = [c for c in df.columns if c not in ['TransactionID', 'TransactionDT', 'isFraud']]
    
    # Shuffle Train Y
    y_train_shuffled = np.random.permutation(df_train['isFraud'].values)
    
    dtrain = xgb.DMatrix(df_train[features], label=y_train_shuffled)
    dtest = xgb.DMatrix(df_test[features])
    
    model = xgb.train({'max_depth': 3, 'objective': 'binary:logistic'}, dtrain, num_boost_round=10)
    preds = model.predict(dtest)
    
    pr_auc = average_precision_score(df_test['isFraud'].values, preds)
    fraud_rate = df_test['isFraud'].mean()
    
    pass_flag = pr_auc < (fraud_rate * 2)  # Generous boundary, should be == fraud rate (~0.03)
    
    return {
        "test_name": "Label Shuffle (v3)",
        "pr_auc": float(pr_auc),
        "target_baseline": float(fraud_rate),
        "pass": bool(pass_flag)
    }

def run_entity_leakage_test():
    """
    Verifies that we are indeed throwing away entities unseen in Phase 2 V3 train 
    rather than magically looking them up.
    """
    logger.info("Running Entity OOV test...")
    df = pd.read_parquet("phase2_v3/artifacts/tx_feature_matrix.parquet")
    with open(f"{ARTIFACTS_DIR}/split_index.json", "r") as f:
        splits = json.load(f)
        
    test_idx = set(splits['test'])
    df_test = df[df['TransactionID'].isin(test_idx)].copy()
    
    # In v3, we generate flags tracking if an entity was explicitly seen in Train
    seen_cols = [c for c in df.columns if c.startswith('entity_seen_')]
    
    if not seen_cols:
        return {"test_name": "Entity Unseen Match", "pass": False, "msg": "No tracking columns found"}
        
    # How often is a transaction totally isolated / OOV for an entity type?
    oov_rates = {}
    for c in seen_cols:
        # seen == 0 means Out Of Vocabulary (OOV)
        oov_rate = (df_test[c] == 0).mean()
        oov_rates[c] = float(oov_rate)
        
    # We WANT OOV to exist. If OOV is literally 0.0% for everything, we probably leaked the entire graph.
    all_zero_oov = all(v == 0.0 for v in oov_rates.values())
    
    return {
        "test_name": "Entity OOV Validity",
        "oov_rates": oov_rates,
        "pass": not all_zero_oov
    }

def integrity_checks():
    logger.info("Running general integrity tests...")
    
    # Sanity checks on collect_predictions logic
    v_preds_files = os.listdir(f"{ARTIFACTS_DIR}/version_preds/")
    if not v_preds_files:
        return {"test_name": "File Existence", "pass": False, "msg": "No predictions found"}
        
    sizes = []
    for vf in v_preds_files:
        df = pd.read_parquet(f"{ARTIFACTS_DIR}/version_preds/{vf}")
        sizes.append(len(df))
        if df['y_score'].isnull().any() or df['y_true'].isnull().any():
            return {"test_name": "NaN Integrity", "pass": False, "msg": f"{vf} contains NaNs."}
            
    if len(set(sizes)) != 1:
         return {"test_name": "Size Match", "pass": False, "msg": "Version rows do not perfectly math."}
         
    return {"test_name": "Join Integrity", "pass": True}

def run_all_tests():
    out = []
    try:
        out.append(run_label_shuffle_test())
    except Exception as e:
        logger.error(f"Failed Label Shuffle: {e}")
        out.append({"test_name": "Label Shuffle (v3)", "pass": False, "error": str(e)})

    try:
        out.append(run_entity_leakage_test())
    except Exception as e:
        logger.error(f"Failed Entity Leak: {e}")
        out.append({"test_name": "Entity OOV Validity", "pass": False, "error": str(e)})
        
    try:
        out.append(integrity_checks())
    except Exception as e:
        logger.error(f"Failed Integrity: {e}")
        out.append({"test_name": "Integrity Match", "pass": False, "error": str(e)})
        
    with open(f"{ARTIFACTS_DIR}/sanity_results.json", "w") as f:
        json.dump(out, f, indent=4)
    logger.info("Finished running core leakage tests.")

if __name__ == "__main__":
    run_all_tests()
