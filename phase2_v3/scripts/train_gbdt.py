import pandas as pd
import numpy as np
import xgboost as xgb
import json
import logging
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = "phase2_v3/artifacts"

def train_xgb():
    logger.info("Loading feature matrix...")
    df = pd.read_parquet(f"{ARTIFACTS_DIR}/tx_feature_matrix.parquet")
    
    with open(f"{ARTIFACTS_DIR}/split_index.json", "r") as f:
        splits = json.load(f)
        
    train_idx = set(splits['train'])
    val_idx = set(splits['val'])
    
    df_train = df[df['TransactionID'].isin(train_idx)].copy()
    df_val = df[df['TransactionID'].isin(val_idx)].copy()
    
    # Drop non-feature columns
    drop_cols = ['TransactionID', 'TransactionDT', 'isFraud']
    features = [c for c in df_train.columns if c not in drop_cols]
    
    logger.info(f"Using {len(features)} features for training.")
    
    X_train, y_train = df_train[features], df_train['isFraud']
    X_val, y_val = df_val[features], df_val['isFraud']
    
    # Calculate scale_pos_weight purely from TRAIN set
    num_pos = y_train.sum()
    num_neg = len(y_train) - num_pos
    scale_weight = num_neg / num_pos if num_pos > 0 else 1.0
    logger.info(f"Scale pos weight based on train: {scale_weight:.2f}")

    # Small data safe XGB params
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
    
    logger.info("Training XGBoost with early stopping...")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=800,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=50
    )
    
    logger.info(f"Best iteration: {model.best_iteration}, Best Val PR-AUC: {model.best_score}")
    
    # Sanity checks
    train_pr = float(model.eval(dtrain).split(":")[1])
    if train_pr > 0.40:
        logger.warning(f"OVERFIT/LEAKAGE ALERT! Train PR-AUC is suspiciously high: {train_pr:.4f}")
        
    # Save Model
    out_model = f"{ARTIFACTS_DIR}/model_xgb.pkl"
    with open(out_model, "wb") as f:
        pickle.dump(model, f)
        
    # Save Configuration
    with open(f"{ARTIFACTS_DIR}/feature_columns.json", "w") as f:
        json.dump(features, f)
        
    train_config = {
        "params": params,
        "best_iteration": model.best_iteration,
        "best_val_pr": model.best_score,
        "num_train": len(X_train),
        "num_val": len(X_val)
    }
    with open(f"{ARTIFACTS_DIR}/train_config.json", "w") as f:
        json.dump(train_config, f)
        
    logger.info("Training complete and artifacts saved.")

if __name__ == "__main__":
    train_xgb()
