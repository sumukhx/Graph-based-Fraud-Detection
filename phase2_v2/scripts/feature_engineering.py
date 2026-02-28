import os
import sys
import time
import logging
import gc
import json
import math
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict, deque

# --- CONFIGURATION ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE2_V2_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append("/home/ubuntu/Graph/graph1/ieee-fraud-detection/phase0")

import build_graph_phase0_v2 as phase0

LOG_FILE = os.path.join(PHASE2_V2_DIR, "logs", "feature_engineering.log")
FEATURES_OUTPUT = os.path.join(PHASE2_V2_DIR, "artifacts", "advanced_features.parquet")

os.makedirs(os.path.join(PHASE2_V2_DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(PHASE2_V2_DIR, "artifacts"), exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[
    logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)
])

# Helpers for entropy
def calc_entropy(counts):
    total = sum(counts)
    if total <= 1: return 0.0
    ent = 0
    for c in counts:
        p = c / total
        if p > 0: ent -= p * math.log2(p)
    return ent

def run():
    logging.info("=== Starting Advanced Feature Engineering ===")
    
    # 1. Load Data
    logging.info("Loading Phase 0 data...")
    df_raw = phase0.load_data()
    df_raw = df_raw.sort_values('TransactionDT').reset_index(drop=True)
    
    # Create target mapping
    tx_fraud_map = dict(zip("TX_" + df_raw['TransactionID'].astype(str), df_raw['isFraud']))
    
    # 2. Build Directed Graph
    logging.info("Building Property Graph...")
    G = phase0.build_graph(df_raw)
    
    # 3. Global Structural Features (Calculated on Undirected for standard metrics)
    logging.info("Calculating Global Node Metrics (Undirected)...")
    G_undirected = G.to_undirected()
    
    logging.info(" - Degree Centrality")
    degree_dict = dict(G_undirected.degree())
    
    # Using pagerank on the directed graph is perfectly fine and often better, but we only use history for fraud stats.
    # Standard topology metrics on the whole graph without fraud labels is considered permissible "unsupervised" learning 
    # similar to node2vec embeddings, provided it doesn't use the 'fraud' label.
    logging.info(" - PageRank")
    pagerank_dict = nx.pagerank(G, alpha=0.85, weight=None) # Keep weight none if not explicitly weighted edges
    
    # Eigenvector centrality can fail to converge on large graphs, using max_iter
    logging.info(" - Eigenvector Centrality")
    try:
        eigen_dict = nx.eigenvector_centrality(G_undirected, max_iter=500, tol=1e-04)
    except:
        logging.warning("Eigenvector failed to converge, defaulting to 0")
        eigen_dict = defaultdict(float)
        
    logging.info(" - Clustering Coefficient")
    # Clustering coefficient requires a simple graph (not MultiGraph)
    G_simple = nx.Graph(G_undirected)
    clustering_dict = nx.clustering(G_simple)
    
    # Betweenness is incredibly slow for 100k+ edges. We will approximate or calculate ego-betweenness.
    # Given time constraints, skipping global betweenness calculation or using a very tiny sample.
    # Instead, we'll calculate ego network metrics per transaction later if strictly needed.
    
    # 4. Temporal Iteration (Strict Time-Safety for Fraud Proximity)
    logging.info("Iterating chronologically to build time-safe features...")
    
    # History tracking
    # entity -> list of (tx_time, tx_node, fraud_label)
    history = defaultdict(list)
    
    features_list = []
    
    start_time = time.time()
    for i, row in df_raw.iterrows():
        if i % 5000 == 0 and i > 0:
            logging.info(f"Processed {i}/{len(df_raw)} transactions...")
            
        t_id = "TX_" + str(row['TransactionID'])
        t_dt = row['TransactionDT']
        is_fraud = row['isFraud']
        
        # Get entities linked to this TX
        linked_entities = list(G.successors(t_id))
        
        # --- A. Entity Diversity & Burst Features ---
        distinct_cards_per_email = set()
        distinct_emails_per_device = set()
        distinct_regions_per_card = set()
        
        # Temporal burst counts (using all linked entities' history)
        tx_5m, tx_1h, tx_6h = 0, 0, 0
        all_past_tx_times = []
        
        # Time-safe Fraud Proximity
        fraud_neighbors_1hop = set()
        fraud_neighbors_2hop = set()
        weighted_fraud_score = 0.0
        
        entity_types = defaultdict(int)
        
        for ent in linked_entities:
            ent_type = G.nodes[ent].get('type', 'Unknown')
            entity_types[ent_type] += 1
            
            past_events = history[ent] # chronological
            
            # Analyze past events
            for p_time, p_tx, p_fraud in reversed(past_events):
                delta_t = t_dt - p_time
                if delta_t < 0: continue # Should never happen due to sorting
                
                # Burst
                all_past_tx_times.append(p_time)
                if delta_t <= 300: tx_5m += 1
                if delta_t <= 3600: tx_1h += 1
                if delta_t <= 21600: tx_6h += 1
                
                # Fraud Proximity (1-hop = TX -> Entity -> Past TX)
                # Since past_events lists TXs linked to this entity, they are explicitly 1-entity-hop away (which is 2-edge hops in bipartite).
                # We'll define "1-hop neighbor transaction" as a TX sharing 1 entity.
                if p_fraud == 1:
                    fraud_neighbors_1hop.add(p_tx)
                    weighted_fraud_score += math.exp(-delta_t / 86400) # Decay over 1 day
                    
                # For entity diversity, we'd theoretically need to pre-map relationships or look them up.
                # To keep it performant, we approximate by looking at the entities linked to past TXs.
                # E.g., if this is an Email, we look at the Card linked to the past TX.
                if ent_type == 'EmailDomain':
                    for past_neighbor in G.successors(p_tx):
                        if G.nodes[past_neighbor].get('type') == 'CardProfile':
                            distinct_cards_per_email.add(past_neighbor)
                elif ent_type == 'Device':
                    for past_neighbor in G.successors(p_tx):
                        if G.nodes[past_neighbor].get('type') == 'EmailDomain':
                            distinct_emails_per_device.add(past_neighbor)
                elif ent_type == 'CardProfile':
                    for past_neighbor in G.successors(p_tx):
                        if G.nodes[past_neighbor].get('type') == 'Region':
                            distinct_regions_per_card.add(past_neighbor)

        # 2-hop logic (TX -> Entity -> Past TX -> Entity -> Older TX) 
        # For simplicity and speed, we will use the global graph to find 2-hop entity paths, BUT strictly filter by time.
        # This is expensive. We'll stick to robust 1-entity-hop fraud interactions which already define strong clusters.
        
        # Compile Row Features
        n_fraud_neighbors = len(fraud_neighbors_1hop)
        total_neighbors = sum(len(history[e]) for e in linked_entities)
        fraud_ratio = n_fraud_neighbors / total_neighbors if total_neighbors > 0 else 0
        
        # Inter-transaction variance
        time_variance = np.var(all_past_tx_times) if len(all_past_tx_times) > 1 else 0.0
        
        # Structure features (summed/max over linked entities for TX vectorization)
        # We also want the TX node's direct structural features
        max_degree = max([degree_dict.get(e, 0) for e in linked_entities] + [0])
        avg_pagerank = np.mean([pagerank_dict.get(e, 0) for e in linked_entities]) if linked_entities else 0
        avg_eigen = np.mean([eigen_dict.get(e, 0) for e in linked_entities]) if linked_entities else 0
        avg_cluster = np.mean([clustering_dict.get(e, 0) for e in linked_entities]) if linked_entities else 0
        
        row_feat = {
            'TransactionID': row['TransactionID'],
            'TransactionDT': t_dt,
            
            # Global Structural
            'tx_degree': degree_dict.get(t_id, 0),
            'max_entity_degree': max_degree,
            'avg_pagerank': avg_pagerank,
            'avg_eigen': avg_eigen,
            'avg_cluster': avg_cluster,
            
            # Diversity
            'dist_cards_per_email': len(distinct_cards_per_email),
            'dist_emails_per_device': len(distinct_emails_per_device),
            'dist_regions_per_card': len(distinct_regions_per_card),
            'entity_entropy': calc_entropy(list(entity_types.values())),
            
            # Time-Safe Fraud Proximity
            'fraud_neighbors_1hop': n_fraud_neighbors,
            'fraud_neighbor_ratio': fraud_ratio,
            'weighted_fraud_score': weighted_fraud_score,
            
            # Temporal Bursts
            'tx_5m': tx_5m,
            'tx_1h': tx_1h,
            'tx_6h': tx_6h,
            'time_variance': time_variance
        }
        features_list.append(row_feat)
        
        # Update History
        # Note: We must update AFTER Feature calculation to prevent self-leakage
        for ent in linked_entities:
            history[ent].append((t_dt, t_id, is_fraud))
            # Keep history manageable (e.g. last 1000 txs per entity)
            if len(history[ent]) > 1000:
                history[ent].pop(0)

    logging.info(f"Feature extraction complete in {time.time() - start_time:.2f} seconds.")
    
    # 5. Save Features
    df_feat = pd.DataFrame(features_list)
    logging.info(f"Saving {len(df_feat)} rows x {len(df_feat.columns)} features to {FEATURES_OUTPUT}")
    df_feat.to_parquet(FEATURES_OUTPUT, index=False)
    
    logging.info("=== Advanced Feature Engineering Complete ===")

if __name__ == "__main__":
    run()
