import pandas as pd
import numpy as np
import xgboost as xgb
import logging
import pickle
import json
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = "test_audit/artifacts/version_preds"
GOLDEN_SPLIT_PATH = "test_audit/artifacts/split_index.json"

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_base_df():
    # Load the base identifiers and true labels for our golden subset (20k)
    df = pd.read_csv("dataset/train_transaction.csv", nrows=20000)
    return df[['TransactionID', 'TransactionDT', 'isFraud']]

def format_and_save(df_preds, version_name, base_df):
    """
    Ensures structural integrity and standardizes columns: 
    [TransactionID, TransactionDT, y_true, y_score]
    """
    if 'y_score' not in df_preds.columns:
        raise ValueError(f"y_score missing in predictions for {version_name}")
        
    # Join with base to enforce strict alignment to the original 20k rows and original labels
    df_merged = base_df.merge(df_preds[['TransactionID', 'y_score']], on='TransactionID', how='inner')
    df_merged.rename(columns={'isFraud': 'y_true'}, inplace=True)
    
    # Integrity Assertions
    assert len(df_merged) > 0, f"No overlapping transactions found for {version_name}!"
    assert not df_merged['y_score'].isnull().any(), f"NaN predictions detected in {version_name}!"
    assert df_merged['TransactionID'].is_unique, f"Duplicate Transactions in {version_name}!"
    
    out_path = f"{ARTIFACTS_DIR}/{version_name}.parquet"
    df_merged.to_parquet(out_path, index=False)
    logger.info(f"Saved standardized predictions for {version_name}: {len(df_merged)} rows.")

def collect_phase1():
    logger.info("Collecting Phase 1 (Heuristics)...")
    try:
        df1 = pd.read_csv("phase1/phase1_scores.csv")
        df1.rename(columns={'risk_score': 'y_score'}, inplace=True)
        return df1
    except FileNotFoundError:
        logger.error("Phase 1 scores not found. Did you run the phase 1 evaluation?")
        return pd.DataFrame()

def collect_phase2_v1():
    logger.info("Collecting Phase 2 V1 (Full Graph Node2Vec)...")
    try:
        # Phase 2 V1 stored an artifacts bundle in `phase2/`
        with open("phase2/artifacts/model_xgb.pkl", "rb") as f:
            model = pickle.load(f)
        with open("phase2/artifacts/feature_columns.json", "r") as f:
            features = json.load(f)
            
        df_feat = pd.read_parquet("phase2/artifacts/tx_feature_matrix.parquet")
        dmat = xgb.DMatrix(df_feat[features])
        preds = model.predict(dmat)
        
        df_out = df_feat[['TransactionID']].copy()
        df_out['y_score'] = preds
        return df_out
    except Exception as e:
        logger.error(f"Failed extracting Phase 2 V1: {e}")
        return pd.DataFrame()

def collect_phase2_v2():
    logger.info("Collecting Phase 2 V2 (Calibrated)...")
    try:
        with open("phase2_v2/artifacts/model_xgb.pkl", "rb") as f:
            model = pickle.load(f)
        with open("phase2_v2/artifacts/feature_columns.json", "r") as f:
            features = json.load(f)
            
        df_feat = pd.read_parquet("phase2_v2/artifacts/tx_feature_matrix.parquet")
        dmat = xgb.DMatrix(df_feat[features])
        preds = model.predict(dmat)
        
        df_out = df_feat[['TransactionID']].copy()
        df_out['y_score'] = preds
        return df_out
    except Exception as e:
        logger.error(f"Failed extracting Phase 2 V2: {e}")
        return pd.DataFrame()

def collect_phase2_v3():
    logger.info("Collecting Phase 2 V3 (Inductive/Baseline)...")
    try:
        with open("phase2_v3/artifacts/model_xgb.pkl", "rb") as f:
            model = pickle.load(f)
        with open("phase2_v3/artifacts/feature_columns.json", "r") as f:
            features = json.load(f)
            
        df_feat = pd.read_parquet("phase2_v3/artifacts/tx_feature_matrix.parquet")
        dmat = xgb.DMatrix(df_feat[features])
        preds = model.predict(dmat)
        
        df_out = df_feat[['TransactionID']].copy()
        df_out['y_score'] = preds
        return df_out
    except Exception as e:
        logger.error(f"Failed extracting Phase 2 V3: {e}")
        return pd.DataFrame()

def run_collection():
    ensure_dir(ARTIFACTS_DIR)
    base_df = load_base_df()
    
    versions = {
        "phase1": collect_phase1(),
        "phase2_v1": collect_phase2_v1(),
        "phase2_v2": collect_phase2_v2(),
        "phase2_v3": collect_phase2_v3()
    }
    
    for v_name, df_preds in versions.items():
        if not df_preds.empty:
            format_and_save(df_preds, v_name, base_df)

if __name__ == "__main__":
    run_collection()
