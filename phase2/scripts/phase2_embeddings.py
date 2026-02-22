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

# --- SET SEEDS ---
np.random.seed(42)
random.seed(42)

# --- CONFIGURATION (Phase 2 Embeddings) ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE2_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
# The phase0 directory with the graph script is under /home/ubuntu/Graph/graph1/ieee-fraud-detection/phase0
sys.path.append("/home/ubuntu/Graph/graph1/ieee-fraud-detection/phase0")

# Import Phase 0 Graph Builder to rebuild the graph
import build_graph_phase0_v2 as phase0

LOG_FILE = os.path.join(PHASE2_DIR, "logs", "phase2_embeddings.log")
EMBEDDING_OUTPUT = os.path.join(PHASE2_DIR, "artifacts", "tx_embeddings.parquet")
MODEL_OUTPUT = os.path.join(PHASE2_DIR, "artifacts", "node2vec.model")

# Node2Vec Hyperparameters (per user instructions)
N2V_PARAMS = {
    "dimensions": 128,
    "walk_length": 30,
    "num_walks": 10,
    "window": 5,
    "min_count": 1,
    "workers": 8,
    "p": 1.0,
    "q": 1.0,
    "seed": 42
}

# --- SETUP LOGGING ---
os.makedirs(os.path.join(PHASE2_DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(PHASE2_DIR, "artifacts"), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

def run():
    logging.info("=== Starting Phase 2: Node2Vec Embedding Generation ===")
    
    # 1. Rebuild Property Graph (Phase 0)
    logging.info("Loading Phase 0 data and rebuilding Property Graph...")
    df_raw = phase0.load_data()
    G_directed = phase0.build_graph(df_raw)
    
    # Mandatory Validation: Graph Consistency
    logging.info(f"Directed Graph Built: {len(G_directed.nodes)} nodes, {len(G_directed.edges)} edges.")
    assert len(G_directed.nodes) == 30935, f"Expected 30935 nodes, got {len(G_directed.nodes)}"
    assert len(G_directed.edges) == 105617, f"Expected 105617 edges, got {len(G_directed.edges)}"
    logging.info("Graph topology verified successfully against Phase-0.")
    
    # 2. Convert to Undirected Graph (Mandatory)
    # This is required because standard node2vec relies on symmetric transitioning; 
    # directed walks can terminate prematurely or fall into isolated sinks, leading to meaningless embeddings.
    logging.info("Converting directed graph to undirected graph explicitly...")
    G = G_directed.to_undirected()
    
    del G_directed, df_raw
    gc.collect()
    
    # 3. Extract TX Nodes
    tx_nodes = [node for node, data in G.nodes(data=True) if data.get('type') == 'Transaction']
    logging.info(f"Targeting {len(tx_nodes)} Transaction (TX) nodes for embedding extraction.")
    
    # 4. Fit Node2Vec
    logging.info(f"Initializing Node2Vec on full graph with {N2V_PARAMS['workers']} workers...")
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
    # Word2Vec window and min_count
    model = node2vec.fit(window=N2V_PARAMS["window"], min_count=N2V_PARAMS["min_count"], batch_words=4)
    
    logging.info(f"Node2Vec Training Complete in {time.time() - start_time:.2f} seconds.")
    
    # 5. Save Embeddings mapped to Transaction IDs
    logging.info("Extracting embeddings for TX nodes...")
    
    embeddings = []
    missing_nodes = 0
    
    for node in tx_nodes:
        if node in model.wv:
            # Drop the 'TX_' prefix to just save the bare TransactionID for joining later
            pure_id = node.replace("TX_", "")
            vec = model.wv[node]
            # Convert to dictionary feature vector {f0: val, f1: val...}
            row = {"TransactionID": int(float(pure_id))} 
            for i, v in enumerate(vec):
                row[f"emb_{i}"] = v
            embeddings.append(row)
        else:
            missing_nodes += 1
            
    if missing_nodes > 0:
        logging.warning(f"Failed to find embeddings for {missing_nodes} TX nodes.")
        
    df_emb = pd.DataFrame(embeddings)
    
    # Save artifacts
    logging.info(f"Saving {len(df_emb)} embeddings to {EMBEDDING_OUTPUT}")
    df_emb.to_parquet(EMBEDDING_OUTPUT, index=False)
    
    logging.info(f"Saving full node2vec model to {MODEL_OUTPUT}")
    model.save(MODEL_OUTPUT)
    
    logging.info("=== Phase 2 Embeddings Generation Complete ===")

if __name__ == "__main__":
    run()
