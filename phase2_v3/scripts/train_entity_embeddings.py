import networkx as nx
from node2vec import Node2Vec
import pandas as pd
import logging
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = "phase2_v3/artifacts"

def train_entity_embeddings():
    logger.info("Loading training graph...")
    import pickle
    with open(f"{ARTIFACTS_DIR}/train_graph.pkl", "rb") as f:
        G_train_directed = pickle.load(f)
    
    logger.info("Converting to undirected graph for random walks...")
    G_train = G_train_directed.to_undirected()
    
    logger.info("Initializing Node2Vec (Stable Defaults for Entities)...")
    # Stable Defaults: dim=128, walk_length=30, num_walks=10, window=5
    node2vec = Node2Vec(G_train, 
                        dimensions=128, 
                        walk_length=30, 
                        num_walks=10, 
                        p=1.0, q=1.0, 
                        workers=4, 
                        seed=42)
    
    logger.info("Fitting Node2Vec model...")
    model = node2vec.fit(window=5, min_count=1, batch_words=4)
    
    logger.info("Extracting embeddings for Entity Nodes ONLY...")
    # Extract only entity embeddings to save space and enforce our deployment contract
    # Our contract: Transactions aggregate their connected entities at runtime.
    entity_nodes = [n for n in G_train.nodes() if not n.startswith("TX_")]
    
    emb_data = []
    for node in entity_nodes:
        emb_vector = model.wv[node]
        row = {"node_id": node}
        for i, val in enumerate(emb_vector):
            row[f"emb_{i}"] = val
        emb_data.append(row)
        
    df_emb = pd.DataFrame(emb_data)
    
    # Determine basic node type
    def get_node_type(node_id):
        return node_id.split("_")[0]

    df_emb['node_type'] = df_emb['node_id'].apply(get_node_type)
    
    output_path = f"{ARTIFACTS_DIR}/entity_embeddings.parquet"
    df_emb.to_parquet(output_path, index=False)
    
    logger.info(f"Saved {len(df_emb)} Entity Embeddings to {output_path}")
    
    # Save config
    config = {
        "dimensions": 128,
        "walk_length": 30,
        "num_walks": 10,
        "seed": 42
    }
    with open(f"{ARTIFACTS_DIR}/embedding_config.json", "w") as f:
        json.dump(config, f)

if __name__ == "__main__":
    train_entity_embeddings()
