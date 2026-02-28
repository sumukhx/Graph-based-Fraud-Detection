import pandas as pd
import json
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = "dataset"
ARTIFACTS_DIR = "test_audit/artifacts"
NUM_ROWS = 20000

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_golden_split():
    logger.info("Generating Single Source of Truth Split...")
    ensure_dir(ARTIFACTS_DIR)
    
    # Load dataset
    df = pd.read_csv(f"{DATA_DIR}/train_transaction.csv", nrows=NUM_ROWS)
    logger.info(f"Loaded {len(df)} transactions.")
    
    # Strict Sort
    df = df.sort_values('TransactionDT', ascending=True).reset_index(drop=True)
    
    # Split lengths
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    # Extract IDs
    train_idx = df['TransactionID'].iloc[:train_end].tolist()
    val_idx = df['TransactionID'].iloc[train_end:val_end].tolist()
    test_idx = df['TransactionID'].iloc[val_end:].tolist()
    
    # Integrity Assertions
    train_set = set(train_idx)
    val_set = set(val_idx)
    test_set = set(test_idx)
    
    assert len(train_set.intersection(val_set)) == 0, "Leakage Alert: Train/Val overlap!"
    assert len(val_set.intersection(test_set)) == 0, "Leakage Alert: Val/Test overlap!"
    assert len(train_set.intersection(test_set)) == 0, "Leakage Alert: Train/Test overlap!"
    
    train_max_dt = df[df['TransactionID'].isin(train_idx)]['TransactionDT'].max()
    val_min_dt = df[df['TransactionID'].isin(val_idx)]['TransactionDT'].min()
    val_max_dt = df[df['TransactionID'].isin(val_idx)]['TransactionDT'].max()
    test_min_dt = df[df['TransactionID'].isin(test_idx)]['TransactionDT'].min()
    
    assert train_max_dt <= val_min_dt, "Leakage Alert: Train time bleeds into Val!"
    assert val_max_dt <= test_min_dt, "Leakage Alert: Val time bleeds into Test!"

    logger.info(f"Audit Splits Verified -> Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Calculate Fraud Rates
    train_fraud = df[df['TransactionID'].isin(train_idx)]['isFraud'].mean()
    val_fraud = df[df['TransactionID'].isin(val_idx)]['isFraud'].mean()
    test_fraud = df[df['TransactionID'].isin(test_idx)]['isFraud'].mean()
    
    logger.info(f"Fraud Rates -> Train: {train_fraud:.4f} | Val: {val_fraud:.4f} | Test: {test_fraud:.4f}")
    
    # Save Split configuration
    split_config = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx
    }
    
    with open(f"{ARTIFACTS_DIR}/split_index.json", "w") as f:
        json.dump(split_config, f)
    logger.info("Golden Split saved to test_audit/artifacts/split_index.json")

if __name__ == "__main__":
    generate_golden_split()
