import os
import sys
import time
import logging
import gc
import random
import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
from sklearn.preprocessing import normalize

# --- SET SEEDS ---
np.random.seed(42)
random.seed(42)

# --- CONFIGURATION (Phase 2 v2 Embeddings) ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE2_V2_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
# The phase0 directory with the graph script is under /home/ubuntu/Graph/graph1/ieee-fraud-detection/phase0
sys.path.append("/home/ubuntu/Graph/graph1/ieee-fraud-detection/phase0")

import build_graph_phase0_v2 as phase0

LOG_FILE = os.path.join(PHASE2_V2_DIR, "logs", "phase2_embeddings_v2.log")
EMBEDDING_OUTPUT = os.path.join(PHASE2_V2_DIR, "artifacts", "tx_embeddings_v2.parquet")

# Node2Vec Hyperparameters (per user instructions)
N2V_PARAMS = {
    "dimensions": 256,
    "walk_length": 40,
    "num_walks": 15,
    "window": 7,
    "min_count": 1,
    "workers": 8,
    "p": 0.5,
    "q": 2.0,
    "seed": 42
}

# --- SETUP LOGGING ---
os.makedirs(os.path.join(PHASE2_V2_DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(PHASE2_V2_DIR, "artifacts"), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

def run():
    logging.info("=== Starting Phase 2 V2: Advanced Node2Vec Embedding Generation ===")
    
    # 1. Rebuild Property Graph (Phase 0)
    logging.info("Loading Phase 0 data and rebuilding Property Graph...")
    df_raw = phase0.load_data()
    G_directed = phase0.build_graph(df_raw)
    
    # Mandatory Validation: Graph Consistency
    logging.info(f"Directed Graph Built: {len(G_directed.nodes)} nodes, {len(G_directed.edges)} edges.")
    
    # 2. Convert to Undirected Graph (Mandatory for optimal node2vec)
    logging.info("Converting directed graph to undirected graph explicitly...")
    G = G_directed.to_undirected()
    
    del G_directed, df_raw
    gc.collect()
    
    # 3. Extract TX Nodes
    tx_nodes = [node for node in G.nodes() if str(node).startswith('TX_')]
    logging.info(f"Targeting {len(tx_nodes)} Transaction (TX) nodes for embedding extraction.")
    
    # 4. Fit Node2Vec
    logging.info(f"Initializing Node2Vec on undirected graph with parameters: {N2V_PARAMS}")
    start_time = time.time()
    
    # Apply strict deterministic seed via np/random libraries before starting
    np.random.seed(42)
    random.seed(42)
    
    node2vec = Node2Vec(G, 
                        dimensions=N2V_PARAMS["dimensions"], 
                        walk_length=N2V_PARAMS["walk_length"], 
                        num_walks=N2V_PARAMS["num_walks"], 
                        workers=N2V_PARAMS["workers"],
                        p=N2V_PARAMS["p"],
                        q=N2V_PARAMS["q"],
                        seed=N2V_PARAMS["seed"],
                        quiet=False)
                        
    logging.info("Node2Vec walks completed. Training Word2Vec model...")
    model = node2vec.fit(window=N2V_PARAMS["window"], min_count=N2V_PARAMS["min_count"], batch_words=4)
    
    logging.info(f"Node2Vec Training Complete in {time.time() - start_time:.2f} seconds.")
    
    # 5. Save and Normalize Embeddings mapped to Transaction IDs
    logging.info("Extracting embeddings for TX nodes...")
    
    row_ids = []
    vectors = []
    missing_nodes = 0
    
    for node in tx_nodes:
        if node in model.wv:
            pure_id = int(float(node.replace("TX_", "")))
            vec = model.wv[node]
            row_ids.append(pure_id)
            vectors.append(vec)
        else:
            missing_nodes += 1
            
    if missing_nodes > 0:
        logging.warning(f"Failed to find embeddings for {missing_nodes} TX nodes.")
    
    # Normalize vectors
    logging.info("Normalizing embeddings (L2 norm)...")
    vectors_array = np.array(vectors)
    normalized_vectors = normalize(vectors_array, norm='l2', axis=1)
    
    # Convert to DataFrame
    df_emb = pd.DataFrame(normalized_vectors, columns=[f"emb_{i}" for i in range(N2V_PARAMS["dimensions"])])
    df_emb.insert(0, "TransactionID", row_ids)
    
    # Save artifacts
    logging.info(f"Saving {len(df_emb)} normalized embeddings to {EMBEDDING_OUTPUT}")
    df_emb.to_parquet(EMBEDDING_OUTPUT, index=False)
    
    logging.info("=== Phase 2 V2 Embeddings Generation Complete ===")

if __name__ == "__main__":
    run()
