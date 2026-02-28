import os
import sys
import time
import logging
import gc
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, confusion_matrix, recall_score, brier_score_loss

# --- CONFIGURATION (Phase 2 V2 Model Training) ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE2_V2_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

LOG_FILE = os.path.join(PHASE2_V2_DIR, "logs", "train_xgboost_v2.log")
FEATURES_INPUT = os.path.join(PHASE2_V2_DIR, "artifacts", "advanced_features.parquet")
EMBEDDINGS_INPUT = os.path.join(PHASE2_V2_DIR, "artifacts", "tx_embeddings_v2.parquet")
PROCESSED_DATA_OUTPUT = os.path.join(PHASE2_V2_DIR, "artifacts", "processed_data.parquet")
MODEL_OUTPUT = os.path.join(PHASE2_V2_DIR, "artifacts", "xgboost_v2.model")

# XGBoost Hyperparameters (per user instructions)
XGB_PARAMS = {
    'max_depth': 8,
    'min_child_weight': 3,
    'gamma': 1,
    'subsample': 0.9,
    'colsample_bytree': 0.8,
    'learning_rate': 0.03,
    'n_estimators': 1000,
    'reg_alpha': 0.5,
    'reg_lambda': 1.0,
    'random_state': 42,
    'eval_metric': 'aucpr',
    'objective': 'binary:logistic',
    'tree_method': 'hist' # faster on modern CPUs
}

os.makedirs(os.path.join(PHASE2_V2_DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(PHASE2_V2_DIR, "artifacts"), exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[
    logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)
])

def run():
    logging.info("=== Starting Phase 2 V2 Model Training ===")
    
    # 1. Load Data
    logging.info(f"Loading features from {FEATURES_INPUT}")
    df_feat = pd.read_parquet(FEATURES_INPUT)
    
    logging.info(f"Loading embeddings from {EMBEDDINGS_INPUT}")
    df_emb = pd.read_parquet(EMBEDDINGS_INPUT)
    
    # Load raw data to get base columns and labels
    sys.path.append("/home/ubuntu/Graph/graph1/ieee-fraud-detection/phase0")
    import build_graph_phase0_v2 as phase0
    df_raw = phase0.load_data()
    df_labels = df_raw[['TransactionID', 'TransactionDT', 'isFraud', 'TransactionAmt']].copy()
    
    # Merge
    logging.info("Merging features, embeddings, and labels...")
    df = df_labels.merge(df_feat, on=['TransactionID', 'TransactionDT'], how='inner')
    df = df.merge(df_emb, on='TransactionID', how='inner')
    
    logging.info(f"Merged Dataset Shape: {df.shape}")
    
    # 2. Feature Interactions & Manual Engineering
    logging.info("Creating manual interaction features...")
    
    # Handle skewed features
    skewed_cols = ['TransactionAmt', 'tx_degree', 'max_entity_degree', 'fraud_neighbors_1hop']
    for col in skewed_cols:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(df[col])
            
    # Interactions
    if 'tx_1h' in df.columns and 'tx_degree' in df.columns:
        df['degree_x_velocity'] = df['tx_degree'] * df['tx_1h']
    
    if 'fraud_neighbor_ratio' in df.columns and 'avg_pagerank' in df.columns:
        df['fraud_ratio_x_pagerank'] = df['fraud_neighbor_ratio'] * df['avg_pagerank']
        
    if 'dist_regions_per_card' in df.columns and 'entity_entropy' in df.columns:
        df['region_collision_x_entropy'] = df['dist_regions_per_card'] * df['entity_entropy']
        
    # Calculate embedding norm manually if we want to interact although we l2 normalized them previously
    # df['emb_norm_x_hub'] = ... (Assuming hub flag isn't explicitly brought up, skip or infer from degree)
    
    # Prepare feature list
    drop_cols = ['TransactionID', 'TransactionDT', 'isFraud']
    features = [c for c in df.columns if c not in drop_cols]
    
    # Standardize continuous features (exclude specific ones if needed, but safe to standardize all base)
    logging.info("Standardizing continuous features...")
    cols_to_scale = [c for c in features if not c.startswith('emb_')]
    scaler = StandardScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    
    # Save processed dataframe for evaluation script
    df.to_parquet(PROCESSED_DATA_OUTPUT, index=False)
    
    # 3. Time-Based Split
    logging.info("Implementing strict time-based split...")
    df = df.sort_values('TransactionDT').reset_index(drop=True)
    n = len(df)
    
    # 14k train, 3k val, 3k test matches user metrics exactly? No, user metrics say 
    # Train: 14000, Val: 3000, Test: 3000. Let's slice exactly.
    train_end = 14000
    val_end = 14000 + 3000
    test_end = 14000 + 3000 + 3000
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:test_end]
    
    X_train, y_train = train_df[features], train_df['isFraud']
    X_val, y_val = val_df[features], val_df['isFraud']
    X_test, y_test = test_df[features], test_df['isFraud']
    
    logging.info(f"Train Split: {X_train.shape[0]} rows, Fraud Rate: {y_train.mean():.4f}")
    logging.info(f"Val Split: {X_val.shape[0]} rows, Fraud Rate: {y_val.mean():.4f}")
    logging.info(f"Test Split: {X_test.shape[0]} rows, Fraud Rate: {y_test.mean():.4f}")
    
    # 4. Handle Imbalance
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    logging.info(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")
    XGB_PARAMS['scale_pos_weight'] = scale_pos_weight
    
    # 5. Train Model
    logging.info("Training XGBoost Model with Early Stopping...")
    XGB_PARAMS['early_stopping_rounds'] = 50
    model = xgb.XGBClassifier(**XGB_PARAMS)
    
    eval_set = [(X_train, y_train), (X_val, y_val)]
    
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=10
    )
    
    logging.info(f"Best Iteration: {model.best_iteration}")
    logging.info(f"Best Score: {model.best_score}")
    
    # Save
    logging.info(f"Saving model to {MODEL_OUTPUT}")
    model.save_model(MODEL_OUTPUT)
    
    logging.info("=== Phase 2 V2 Model Training Complete ===")

if __name__ == "__main__":
    run()
