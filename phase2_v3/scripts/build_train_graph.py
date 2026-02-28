import pandas as pd
import networkx as nx
import numpy as np
import json
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "dataset"
ARTIFACTS_DIR = "phase2_v3/artifacts"
NUM_ROWS = 20000

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_and_split_data(num_rows):
    logger.info(f"Loading top {num_rows} rows from dataset...")
    
    # Load Transactions
    df_tx = pd.read_csv(f"{DATA_DIR}/train_transaction.csv", nrows=num_rows)
    
    # Load Identity (only for matching transactions)
    df_id = pd.read_csv(f"{DATA_DIR}/train_identity.csv")
    df_id = df_id[df_id['TransactionID'].isin(df_tx['TransactionID'])]
    
    # Merge
    df = df_tx.merge(df_id, on='TransactionID', how='left')
    logger.info(f"Merged Data Shape: {df.shape}")
    
    # Sort strictly by time
    df = df.sort_values('TransactionDT', ascending=True).reset_index(drop=True)
    
    # Split indices (70/15/15)
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    train_idx = df['TransactionID'].iloc[:train_end].tolist()
    val_idx = df['TransactionID'].iloc[train_end:val_end].tolist()
    test_idx = df['TransactionID'].iloc[val_end:].tolist()
    
    # Overly strict time split check
    train_max_dt = df[df['TransactionID'].isin(train_idx)]['TransactionDT'].max()
    val_min_dt = df[df['TransactionID'].isin(val_idx)]['TransactionDT'].min()
    val_max_dt = df[df['TransactionID'].isin(val_idx)]['TransactionDT'].max()
    test_min_dt = df[df['TransactionID'].isin(test_idx)]['TransactionDT'].min()
    
    assert train_max_dt <= val_min_dt, "Leakage Alert: Train time overlaps with Val time!"
    assert val_max_dt <= test_min_dt, "Leakage Alert: Val time overlaps with Test time!"
    
    logger.info(f"Splits generated -> Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Save Split configuration
    split_config = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx
    }
    
    with open(f"{ARTIFACTS_DIR}/split_index.json", "w") as f:
        json.dump(split_config, f)
        
    # Return ONLY the Train portion for graph building
    df_train = df[df['TransactionID'].isin(train_idx)].copy()
    return df_train

def build_train_only_graph(df_train):
    logger.info("Building Directed Heterogeneous Graph ONLY from Train split...")
    G = nx.DiGraph()
    
    # Extract structural edges according to Phase-0 schema
    # (TX -> Entities)
    
    edges_to_add = []
    
    # Cards
    cards = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6']
    for c in cards:
        valid_df = df_train[df_train[c].notnull()]
        for _, row in valid_df.iterrows():
            entity_id = f"{c}_{row[c]}"
            edges_to_add.append(("TX_" + str(row['TransactionID']), "Card_" + entity_id))
            
    # Emails
    for e in ['P_emaildomain', 'R_emaildomain']:
        valid_df = df_train[df_train[e].notnull()]
        for _, row in valid_df.iterrows():
            entity_id = f"Email_{row[e]}"
            edges_to_add.append(("TX_" + str(row['TransactionID']), entity_id))
            
    # Devices
    valid_df = df_train[df_train['DeviceInfo'].notnull()]
    for _, row in valid_df.iterrows():
        entity_id = f"Device_{row['DeviceInfo']}"
        edges_to_add.append(("TX_" + str(row['TransactionID']), entity_id))
        
    # Address (Region/Addr)
    for a in ['addr1', 'addr2']:
        valid_df = df_train[df_train[a].notnull()]
        for _, row in valid_df.iterrows():
            entity_id = f"Addr_{a}_{row[a]}"
            edges_to_add.append(("TX_" + str(row['TransactionID']), entity_id))
            
    # Identity Profile (id_31 browser, id_33 resolution)
    for id_col in ['id_31', 'id_33']:
        if id_col in df_train.columns:
            valid_df = df_train[df_train[id_col].notnull()]
            for _, row in valid_df.iterrows():
                entity_id = f"IDP_{id_col}_{row[id_col]}"
                edges_to_add.append(("TX_" + str(row['TransactionID']), entity_id))

    G.add_edges_from(edges_to_add)
    
    logger.info(f"Graph constructed: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    
    # Node Typology Check (Approximate)
    tx_nodes = [n for n in G.nodes if n.startswith("TX_")]
    entity_nodes = [n for n in G.nodes if not n.startswith("TX_")]
    logger.info(f"Typology -> Transaction Nodes: {len(tx_nodes)} | Entity Nodes: {len(entity_nodes)}")
    
    # Save train graph
    import pickle
    with open(f"{ARTIFACTS_DIR}/train_graph.pkl", "wb") as f:
        pickle.dump(G, f)
    logger.info(f"Train graph saved to {ARTIFACTS_DIR}/train_graph.pkl")

if __name__ == "__main__":
    ensure_dir(ARTIFACTS_DIR)
    df_train = load_and_split_data(NUM_ROWS)
    build_train_only_graph(df_train)
