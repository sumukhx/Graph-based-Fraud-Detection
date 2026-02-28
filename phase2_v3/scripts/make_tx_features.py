import pandas as pd
import numpy as np
import logging
import json
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = "phase2_v3/artifacts"
DATA_DIR = "dataset"
NUM_ROWS = 20000

def load_data():
    logger.info("Loading transaction and identity data...")
    df_tx = pd.read_csv(f"{DATA_DIR}/train_transaction.csv", nrows=NUM_ROWS)
    df_id = pd.read_csv(f"{DATA_DIR}/train_identity.csv")
    df_id = df_id[df_id['TransactionID'].isin(df_tx['TransactionID'])]
    df = df_tx.merge(df_id, on='TransactionID', how='left')
    df = df.sort_values('TransactionDT', ascending=True).reset_index(drop=True)
    
    with open(f"{ARTIFACTS_DIR}/split_index.json", "r") as f:
        splits = json.load(f)
        
    return df, splits

def create_embeddings_features(df, splits):
    logger.info("Loading entity embeddings...")
    df_emb = pd.read_parquet(f"{ARTIFACTS_DIR}/entity_embeddings.parquet")
    
    emb_dim = len([c for c in df_emb.columns if c.startswith("emb_")])
    logger.info(f"Loaded embeddings for {len(df_emb)} entities with dim={emb_dim}")
    
    # Create lookup dict: node_id -> numpy array
    emb_keys = [c for c in df_emb.columns if c.startswith("emb_")]
    emb_dict = df_emb.set_index('node_id')[emb_keys].apply(tuple, axis=1).to_dict()
    emb_dict = {k: np.array(v) for k, v in emb_dict.items()}
    
    # Initialize arrays for fast assignment
    n_rows = len(df)
    emb_mean = np.zeros((n_rows, emb_dim))
    emb_max = np.zeros((n_rows, emb_dim))
    emb_std = np.zeros((n_rows, emb_dim))
    
    # Missing flags
    missing_flags = {
        'entity_seen_in_train_device': np.zeros(n_rows, dtype=int),
        'entity_seen_in_train_email': np.zeros(n_rows, dtype=int),
        'entity_seen_in_train_card': np.zeros(n_rows, dtype=int),
        'entity_seen_in_train_addr': np.zeros(n_rows, dtype=int),
        'entity_seen_in_train_idp': np.zeros(n_rows, dtype=int),
    }

    logger.info("Aggregating entity embeddings per transaction...")
    for idx, row in df.iterrows():
        tx_entities = []
        
        # Cards
        for c in ['card1', 'card2', 'card3', 'card4', 'card5', 'card6']:
            if pd.notnull(row[c]):
                tx_entities.append((f"Card_{c}_{row[c]}", 'card'))
        
        # Emails
        for e in ['P_emaildomain', 'R_emaildomain']:
            if pd.notnull(row[e]):
                tx_entities.append((f"Email_{row[e]}", 'email'))
                
        # Device
        if pd.notnull(row.get('DeviceInfo')):
            tx_entities.append((f"Device_{row['DeviceInfo']}", 'device'))
            
        # Address
        for a in ['addr1', 'addr2']:
            if pd.notnull(row[a]):
                tx_entities.append((f"Addr_{a}_{row[a]}", 'addr'))
                
        # Identity
        for id_col in ['id_31', 'id_33']:
            if pd.notnull(row.get(id_col)):
                tx_entities.append((f"IDP_{id_col}_{row[id_col]}", 'idp'))
                
        # Fetch embeddings
        valid_embs = []
        for e_id, e_type in tx_entities:
            if e_id in emb_dict:
                valid_embs.append(emb_dict[e_id])
                missing_flags[f'entity_seen_in_train_{e_type}'][idx] = 1
                
        if valid_embs:
            stack = np.vstack(valid_embs)
            emb_mean[idx] = np.mean(stack, axis=0)
            emb_max[idx] = np.max(stack, axis=0)
            emb_std[idx] = np.std(stack, axis=0)
            
    # Combine into dataframe
    df_mean = pd.DataFrame(emb_mean, columns=[f"emb_mean_{i}" for i in range(emb_dim)])
    df_max = pd.DataFrame(emb_max, columns=[f"emb_max_{i}" for i in range(emb_dim)])
    df_std = pd.DataFrame(emb_std, columns=[f"emb_std_{i}" for i in range(emb_dim)])
    df_flags = pd.DataFrame(missing_flags)
    
    return pd.concat([df_mean, df_max, df_std, df_flags], axis=1)

def build_time_safe_struct(df, splits):
    logger.info("Building strictly time-safe structural features...")
    # NOTE: In a true streaming setup, this rolling count updates row by row.
    # Here, pandas strictly preserves time-order.
    
    n_rows = len(df)
    device_rolling_count = np.zeros(n_rows)
    device_state = {}
    
    train_idx_set = set(splits['train'])
    
    # Build structural degree lookup only from train.
    # Count frequency of entities in train set only!
    train_entities = {}
    for idx, row in df.iterrows():
        if row['TransactionID'] in train_idx_set:
            if pd.notnull(row.get('DeviceInfo')):
                d_id = f"Device_{row['DeviceInfo']}"
                train_entities[d_id] = train_entities.get(d_id, 0) + 1
            for e in ['P_emaildomain', 'R_emaildomain']:
                if pd.notnull(row[e]):
                    e_id = f"Email_{row[e]}"
                    train_entities[e_id] = train_entities.get(e_id, 0) + 1
                    
    degree_train = np.zeros(n_rows)
    hub_flag = np.zeros(n_rows, dtype=int)
    
    for idx, row in df.iterrows():
        # Rolling count (time-safe)
        dev = row.get('DeviceInfo')
        if pd.notnull(dev):
            count = device_state.get(dev, 0)
            device_rolling_count[idx] = count
            # DO update state even in Val/Test because it only uses past TXs (no labels!)
            device_state[dev] = count + 1
            
        # Train degree
        max_deg = 0
        if pd.notnull(dev):
            max_deg = max(max_deg, train_entities.get(f"Device_{dev}", 0))
        for e in ['P_emaildomain', 'R_emaildomain']:
            if pd.notnull(row[e]):
                max_deg = max(max_deg, train_entities.get(f"Email_{row[e]}", 0))
                
        degree_train[idx] = max_deg
        if max_deg >= 500:
            hub_flag[idx] = 1

    ans = pd.DataFrame({
        'struct_device_rolling_count': device_rolling_count,
        'struct_train_degree_max': degree_train,
        'struct_train_hub_flag': hub_flag
    })
    return ans

def make_tabular_features(df):
    logger.info("Generating minimal tabular features...")
    df_raw = pd.DataFrame()
    df_raw['raw_log_TransactionAmt'] = np.log1p(df['TransactionAmt'])
    
    # Categoricals
    df_raw['raw_ProductCD'] = pd.factorize(df['ProductCD'])[0]
    
    df_raw['raw_dist1'] = df['dist1'].fillna(-1)
    df_raw['raw_dist2'] = df['dist2'].fillna(-1)
    
    c_cols = [f'C{i}' for i in range(1, 15)]
    d_cols = [f'd{i}' for i in range(1, 16)]
    d_cols = [col for col in d_cols if col in df.columns]
    
    df_raw['raw_C_mean'] = df[c_cols].mean(axis=1).fillna(0)
    df_raw['raw_D_mean'] = df[d_cols].mean(axis=1).fillna(-1)
    df_raw['raw_C_null_count'] = df[c_cols].isnull().sum(axis=1)
    df_raw['raw_D_null_count'] = df[d_cols].isnull().sum(axis=1)
    
    return df_raw

def main():
    df, splits = load_data()
    
    # Base columns
    df_base = df[['TransactionID', 'TransactionDT', 'isFraud']].copy()
    
    # Phase 1 Scores (Join)
    logger.info("Joining Phase-1 scores...")
    df_phase1 = pd.read_csv("phase1/phase1_scores.csv")
    df_base = df_base.merge(df_phase1[['TransactionID', 'risk_score', 'H1_score', 'H2_score', 'H3_score', 'H4_score', 'H5_score', 'H6_score']], 
                            on='TransactionID', how='left')
    
    # Rename to phase1_*
    df_base.rename(columns={
        'risk_score': 'phase1_risk_score',
        'H1_score': 'phase1_H1', 'H2_score': 'phase1_H2', 'H3_score': 'phase1_H3',
        'H4_score': 'phase1_H4', 'H5_score': 'phase1_H5', 'H6_score': 'phase1_H6'
    }, inplace=True)
    df_base.fillna({'phase1_risk_score': 0, 'phase1_H1': 0, 'phase1_H2': 0, 'phase1_H3': 0, 'phase1_H4': 0, 'phase1_H5': 0, 'phase1_H6': 0}, inplace=True)
    
    df_embs = create_embeddings_features(df, splits)
    df_struct = build_time_safe_struct(df, splits)
    df_raw = make_tabular_features(df)
    
    df_final = pd.concat([df_base, df_embs, df_struct, df_raw], axis=1)
    
    # Type check for assertions
    num_cols = df_final.shape[1]
    logger.info(f"Final feature matrix shape: {df_final.shape}")
    
    assert num_cols < 2000, "Leakage/Bloat Alert: Too many features (>2000) generated!"
    
    # Save
    out_path = f"{ARTIFACTS_DIR}/tx_feature_matrix.parquet"
    df_final.to_parquet(out_path, index=False)
    logger.info(f"Feature matrix saved to {out_path}")

if __name__ == "__main__":
    main()
