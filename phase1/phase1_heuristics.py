
import sys
import os
import pandas as pd
import numpy as np
import networkx as nx
import json
import time
from collections import defaultdict, deque, Counter
from datetime import timedelta
import math

try:
    from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, recall_score
    SKLEARN_AVAIL = True
except ImportError:
    SKLEARN_AVAIL = False
    print("WARNING: sklearn not found, evaluation metrics will be limited.")

# --- PATH SETUP ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE0_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "phase0"))
sys.path.append(PHASE0_DIR)

# Import Phase 0 Graph Builder
import build_graph_phase0_v2 as phase0

# --- CONFIGURATION (UPDATED PER USER REQUEST) ---
OUTPUT_FILE = os.path.join(CURRENT_DIR, "phase1_scores.csv")

# ⏱️ Time windows (TransactionDT is in seconds)
TIME_WINDOWS = {
    "short": 3600,        # 1 hour
    "medium": 21600,      # 6 hours
    "long": 86400         # 24 hours
}

# 🔁 Recency decay (exponential)
RECENCY_TAU = 7200       # 2 hours

# 🌐 Hub control
HUB_DEGREE_THRESHOLD = 500
HUB_NEIGHBOR_CAP = 50
HUB_DOWNWEIGHT_METHOD = "cap"   # options: "cap" or "sqrt"

# 🚨 Heuristic thresholds
# H1 — Shared Device Burst
H1_DEVICE_COUNT_THRESHOLD = 10      # tx in window
H1_TIME_WINDOW = TIME_WINDOWS["short"]

# H2 — Card Issuer + Region collision
H2_MIN_DISTINCT_REGIONS = 3
H2_MIN_DISTINCT_ISSUERS = 3
H2_TIME_WINDOW = TIME_WINDOWS["medium"]

# H3 — Email Domain reuse
H3_MIN_DISTINCT_CARDS = 8
H3_MIN_DISTINCT_DEVICES = 5
H3_TIME_WINDOW = TIME_WINDOWS["medium"]

# H4 — Identity reuse / rarity
H4_RARITY_GLOBAL_FREQ = 5          # rare if < 5 total occurrences
H4_REUSE_THRESHOLD = 3             # reused across ≥3 entities
H4_TIME_WINDOW = TIME_WINDOWS["medium"]

# H5 — Local 2-hop density (ring score)
H5_MIN_SHARED_ENTITIES = 2          # e.g. device + email
H5_MIN_NEIGHBOR_TX = 6
H5_TIME_WINDOW = TIME_WINDOWS["long"]

# H6 — Velocity / rapid hops
H6_MAX_DELTA_T = 300                # 5 minutes
H6_MIN_CONSECUTIVE_TX = 4

# ⚖️ Heuristic weights (must sum ≤ 1)
# NOTE: User provided key names differ slightly from implementation (H1 vs H1_shared_device).
# We will map them.
HEURISTIC_WEIGHTS = {
    "H1": 0.22, # H1_shared_device
    "H2": 0.15, # H2_card_region
    "H3": 0.15, # H3_email_reuse
    "H4": 0.18, # H4_identity
    "H5": 0.20, # H5_density
    "H6": 0.10  # H6_velocity
}

# 🚦 Risk buckets
RISK_THRESHOLDS = {
    "low": 0.30,
    "medium": 0.60,
    "high": 0.80
}

# 🧠 Explainability limits
MAX_TOP_REASONS = 3
MAX_EVIDENCE_ENTITIES = 5


# --- SCORER ---

class FraudScorer:
    def __init__(self, G):
        self.G = G
        # History: EntityID -> Deque of {'t': transaction_dt, 'tx': transaction_id, 'entities': {type: [ids]}}
        self.history = defaultdict(deque)
        
        # Identity Global Counts (Simulated for rarity check)
        # In a real streaming system, this would be an approximate count sketch.
        # Here we can pre-calculate global counts from Phase 0 or build incrementally.
        # The prompt says "H4_RARITY_GLOBAL_FREQ = 5". 
        # Ideally, we should check "count so far".
        self.identity_counts = defaultdict(int)
        
        self.scores = []

    def get_time_weight(self, dt, current_dt):
        delta = current_dt - dt
        if delta < 0: return 0.0 # Should not happen if sorted
        return math.exp(-delta / RECENCY_TAU)

    def get_past_window(self, entity_id, current_dt, window, cap=None):
        """
        Get past transactions for an entity within window.
        Returns list of event dicts.
        """
        dq = self.history[entity_id]
        
        min_time = current_dt - window
        
        relevant = []
        # Iterate from right (most recent)
        for i in range(len(dq) - 1, -1, -1):
            ev = dq[i]
            if ev['t'] < min_time:
                break
            relevant.append(ev)
            if cap and len(relevant) >= cap:
                break
                
        return relevant

    def compute_heuristics(self, tx_id, current_dt, current_entities):
        """
        Compute H1-H6 scores using NEW User Parameters.
        """
        # Unwrap entities
        devices = current_entities.get('Device', [])
        issuers = current_entities.get('CardIssuer', [])
        regions = current_entities.get('Region', [])
        emails = current_entities.get('EmailDomain', [])
        identities = current_entities.get('IdentityProfile', [])
        card_profiles = current_entities.get('CardProfile', [])
        
        reasons = []
        evidence = {}
        
        # --- H1: Shared Device Burst ---
        # "TX is risky if its Device connects to many recent Transactions within ΔT."
        s_h1 = 0.0
        for dev in devices:
            # Check history
            past = self.get_past_window(dev, current_dt, H1_TIME_WINDOW, cap=HUB_NEIGHBOR_CAP)
            if not past: continue
            
            # Count weighted
            w_count = sum(self.get_time_weight(x['t'], current_dt) for x in past)
            
            # User Threshold: H1_DEVICE_COUNT_THRESHOLD = 10
            # If w_count approaches or exceeds 10, risk is high.
            # Sigmoid scaling: 1.0 at threshold.
            
            if w_count >= H1_DEVICE_COUNT_THRESHOLD:
                raw_score = 1.0
            else:
                raw_score = w_count / H1_DEVICE_COUNT_THRESHOLD
                
            if raw_score > s_h1:
                s_h1 = raw_score
                if raw_score > 0.3:
                    evidence['H1'] = f"Device {dev} used in {len(past)} recent TXs (Weighted={w_count:.1f})"
        
        # --- H2: Card Issuer + Region Collision ---
        # "TX is risky if the same CARD1 (issuer) appears across multiple Regions in a short window"
        # OR "same Region with multiple CardIssuers"
        s_h2 = 0.0
        
        # 2a. Issuer -> Many Regions
        for iss in issuers:
            past = self.get_past_window(iss, current_dt, H2_TIME_WINDOW, cap=100)
            seen_regions = set()
            for x in past:
                 seen_regions.update(x['entities'].get('Region', []))
            
            cnt = len(seen_regions)
            if cnt >= H2_MIN_DISTINCT_REGIONS:
                # Scale: if == min, score 0.5? if >> min score 1.0?
                # Let's map Min -> 0.5, Min*2 -> 1.0
                sc = min(cnt / (H2_MIN_DISTINCT_REGIONS * 2), 1.0)
                # Boost if really high
                if cnt >= H2_MIN_DISTINCT_REGIONS: sc = max(sc, 0.5) 
                if cnt >= H2_MIN_DISTINCT_REGIONS * 2: sc = 1.0
                
                if sc > s_h2: 
                    s_h2 = sc
                    reasons.append(f"Issuer {iss} linked to {cnt} Regions")
        
        # 2b. Region -> Many Issuers
        for reg in regions:
             # Check Hub control
             if self.G.nodes[reg].get('is_hub'): 
                 if HUB_DOWNWEIGHT_METHOD == "cap": 
                     pass # handled by get_past_window cap if implemented, but we manually cap below
                 else:
                     continue # skip if strict

             past = self.get_past_window(reg, current_dt, H2_TIME_WINDOW, cap=HUB_NEIGHBOR_CAP)
             seen_iss = set()
             for x in past:
                 seen_iss.update(x['entities'].get('CardIssuer', []))
             
             cnt = len(seen_iss)
             if cnt >= H2_MIN_DISTINCT_ISSUERS:
                 sc = min(cnt / (H2_MIN_DISTINCT_ISSUERS * 2), 1.0)
                 if cnt >= H2_MIN_DISTINCT_ISSUERS: sc = max(sc, 0.5)
                 if cnt >= H2_MIN_DISTINCT_ISSUERS * 2: sc = 1.0
                 
                 if sc > s_h2: s_h2 = sc
        
        if s_h2 > 0.4 and not evidence.get('H2'): 
             evidence['H2'] = "High Variation in Issuer-Region pairs"

        # --- H3: Email Domain Reuse ---
        # "P_EMAIL or R_EMAIL domain links to many distinct Cards/Devices recently"
        s_h3 = 0.0
        for em in emails:
            is_hub = self.G.nodes[em].get('is_hub')
            if is_hub and HUB_DOWNWEIGHT_METHOD == "sqrt":
                 # Skip or heavily downweight
                 continue 
            
            cap = HUB_NEIGHBOR_CAP if is_hub else 200
            past = self.get_past_window(em, current_dt, H3_TIME_WINDOW, cap=cap)
            
            seen_devs = set()
            seen_cards = set()
            for x in past:
                seen_devs.update(x['entities'].get('Device', []))
                seen_cards.update(x['entities'].get('CardProfile', []))
            
            c_cnt = len(seen_cards)
            d_cnt = len(seen_devs)
            
            # Check thresholds
            score_em = 0.0
            if c_cnt >= H3_MIN_DISTINCT_CARDS:
                score_em = max(score_em, min(c_cnt / (H3_MIN_DISTINCT_CARDS * 1.5), 1.0))
            if d_cnt >= H3_MIN_DISTINCT_DEVICES:
                score_em = max(score_em, min(d_cnt / (H3_MIN_DISTINCT_DEVICES * 1.5), 1.0))

            if score_em > s_h3:
                s_h3 = score_em
                if score_em > 0.3:
                     evidence['H3'] = f"Email {em} used with {c_cnt} cards, {d_cnt} devices"

        # --- H4: Identity Profile Rarity / Reuse ---
        # "Risk if IdentityProfile is rare globally but suddenly reused"
        s_h4 = 0.0
        if identities:
            for pid in identities:
                # 1. Check Global Rarity (Count up to now)
                # self.identity_counts updated in update_history or here?
                # We'll update iteratively. CURRENT count includes this TX? 
                # Ideally, look at count BEFORE this tx.
                curr_global_count = self.identity_counts[pid] # This is count PRIOR to update_history of current
                
                # If rare (< 5)
                if curr_global_count < H4_RARITY_GLOBAL_FREQ:
                    # Check reuse in short window
                    past = self.get_past_window(pid, current_dt, H4_TIME_WINDOW)
                    
                    # Distinct Cards or Devices or just raw count?
                    # "reused across >= 3 entities" -> Interpretation: tied to 3 different cards/devices/txs?
                    # Let's count distinct CardProfiles
                    seen_cards = set()
                    for x in past:
                        seen_cards.update(x['entities'].get('CardProfile', []))
                    
                    # If this TX brings a new card, count +1?
                    # The set 'seen_cards' is from PAST.
                    # Does current TX use a card?
                    current_cards = current_entities.get('CardProfile', [])
                    all_cards = seen_cards.copy()
                    all_cards.update(current_cards)
                    
                    reuse_metric = len(all_cards)
                    
                    if reuse_metric >= H4_REUSE_THRESHOLD:
                        # Rare identity used on 3+ cards suddenly
                        s_h4 = 1.0
                        evidence['H4'] = f"Rare Identity {pid} (Freq={curr_global_count}) on {reuse_metric} Cards"
                else:
                     # Not rare, maybe just a common ID. But user rule emphasizes "Rare but reused".
                     pass
        
        # If no explicit ID, H4 is 0.

        # --- H5: Local Graph Density / Ring Score ---
        # "Within 2-hop neighborhood... unusually high transaction overlap"
        s_h5 = 0.0
        overlap_counts = Counter()
        
        for etype, elist in current_entities.items():
            for e in elist:
                # Skip hubs for density check
                if self.G.nodes[e].get('is_hub'): continue
                
                past = self.get_past_window(e, current_dt, H5_TIME_WINDOW)
                for x in past:
                    overlap_counts[x['tx']] += 1
                    
        # Metric: Number of 'neighbor transactions' that share >= MIN_SHARED_ENTITIES (2) with current
        dense_neighbors = 0
        for ptx, count in overlap_counts.items():
            if count >= H5_MIN_SHARED_ENTITIES:
                dense_neighbors += 1
        
        # Threshold: H5_MIN_NEIGHBOR_TX = 6
        if dense_neighbors >= H5_MIN_NEIGHBOR_TX:
            s_h5 = 1.0
        elif dense_neighbors > 1:
            s_h5 = dense_neighbors / H5_MIN_NEIGHBOR_TX
            
        if s_h5 > 0.3:
             evidence['H5'] = f"{dense_neighbors} past TXs share >={H5_MIN_SHARED_ENTITIES} entities"

        # --- H6: Velocity / Rapid Hops ---
        # "H6_MAX_DELTA_T = 300, H6_MIN_CONSECUTIVE_TX = 4"
        s_h6 = 0.0
        targets = card_profiles + devices
        for t in targets:
            past = self.get_past_window(t, current_dt, H6_MAX_DELTA_T * H6_MIN_CONSECUTIVE_TX, cap=H6_MIN_CONSECUTIVE_TX+1)
            # Need at least (N-1) past transactions to form a chain of N including current
            # Check consecutive time diffs?
            # Or just "Count in last X seconds" >= N
            
            # Simple interpretation: If >= X transactions in last Y seconds
            # User says: "Very small Delta T"
            # And "H6_MIN_CONSECUTIVE_TX = 4"
            
            # Let's count events in last (MIN_CONSECUTIVE * MAX_DELTA_T) window?
            # Or strict "each hop < 300s".
            
            cnt = 0
            if past:
                # Check times. 
                # Past is list of events.
                # Check recent burst.
                last_time = current_dt
                chain_len = 1 # Current is 1
                
                # Check backwards from most recent past
                for ev in past:
                     if last_time - ev['t'] < H6_MAX_DELTA_T:
                         chain_len += 1
                         last_time = ev['t']
                     else:
                         break
                
                if chain_len >= H6_MIN_CONSECUTIVE_TX:
                    s_h6 = 1.0
                    evidence['H6'] = f"Velocity: {chain_len} TXs in rapid sequence (<{H6_MAX_DELTA_T}s gaps)"
                    break

        # --- Aggregation ---
        scores = { 'H1': s_h1, 'H2': s_h2, 'H3': s_h3, 'H4': s_h4, 'H5': s_h5, 'H6': s_h6 }
        
        p_safe = 1.0
        top_reasons = []
        
        for k, s in scores.items():
            w = HEURISTIC_WEIGHTS.get(k, 0.1)
            term = 1.0 - (w * s)
            term = max(0.0, min(1.0, term))
            p_safe *= term
            
            if s > 0.0:
                # Add reason
                desc = evidence.get(k, f"{k} Score={s:.2f}")
                # Store (Impact, String)
                top_reasons.append((s * w, f"{k}: {desc}"))
                
        final_risk = 1.0 - p_safe
        
        # Format Top Reasons
        top_reasons.sort(key=lambda x: x[0], reverse=True)
        reason_str = "; ".join([x[1] for x in top_reasons[:MAX_TOP_REASONS]])
        
        # Cap Evidence Entities
        # Already JSON, but let's ensure evidence dict isn't huge? 
        # The evidence dict keys are H1..H6, values are strings. It's fine.
        
        return final_risk, scores, reason_str, evidence

    def update_history(self, tx_id, tx_dt, current_entities):
        event = {
            't': tx_dt,
            'tx': tx_id,
            'entities': current_entities
        }
        for etype, elist in current_entities.items():
            for e in elist:
                self.history[e].append(event)
                # Prune
                if len(self.history[e]) > 2000: 
                    self.history[e].popleft()
        
        # Update Identity Counts
        ids = current_entities.get('IdentityProfile', [])
        for pid in ids:
            self.identity_counts[pid] += 1

    def run(self):
        print("Starting Scoring Engine...")
        
        tx_nodes = []
        for n, d in self.G.nodes(data=True):
            if d.get('type') == 'Transaction':
                dt = d.get('dt', 0)
                tx_nodes.append((n, dt))
        
        tx_nodes.sort(key=lambda x: x[1])
        print(f"Processing {len(tx_nodes)} transactions in time order...")
        
        results = []
        start_time = time.time()
        
        for i, (tx_id, dt) in enumerate(tx_nodes):
            if i % 2000 == 0:
                print(f"Processed {i}/{len(tx_nodes)}...")
                
            current_entities = defaultdict(list)
            for neighbor in self.G.successors(tx_id):
                ntype = self.G.nodes[neighbor].get('type')
                current_entities[ntype].append(neighbor)
            
            risk, scores, reason, ev = self.compute_heuristics(tx_id, dt, current_entities)
            
            row = {
                'TransactionID': tx_id.replace('TX_', ''),
                'TransactionDT': dt,
                'risk_score': round(risk, 4),
                'top_reasons': reason,
                'evidence_json': json.dumps(ev),
                'isFraud': self.G.nodes[tx_id].get('fraud') 
            }
            for k, s in scores.items():
                row[f"{k}_score"] = round(s, 4)
                
            results.append(row)
            
            self.update_history(tx_id, dt, current_entities)
            
        print(f"Scoring complete in {time.time() - start_time:.2f}s")
        return pd.DataFrame(results)

def evaluate(df):
    print("\n--- EVALUATION (Using isFraud for Validation ONLY) ---")
    if 'isFraud' not in df.columns:
        print("No Ground Truth available.")
        return
        
    y_true = df['isFraud'].fillna(0)
    y_score = df['risk_score']
    
    if SKLEARN_AVAIL:
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            prauc = auc(recall, precision)
            roc = roc_auc_score(y_true, y_score)
            print(f"PR-AUC: {prauc:.4f}")
            print(f"ROC-AUC: {roc:.4f}")
            
            print("\nRisk Buckets analysis:")
            # Bucket Definition
            # "low": 0.30, "medium": 0.60, "high": 0.80
            def get_bucket(s):
                if s < RISK_THRESHOLDS['low']: return 'Low'
                if s < RISK_THRESHOLDS['medium']: return 'Medium'
                if s < RISK_THRESHOLDS['high']: return 'High'
                return 'Critical'
                
            df['bucket'] = df['risk_score'].apply(get_bucket)
            print(df.groupby('bucket')['isFraud'].agg(['count', 'mean']))
            
        except Exception as e:
            print(f"Metric calculation failed: {e}")
            
    print("\n--- TOP 10 RISKIEST TRANSACTIONS ---")
    top10 = df.sort_values('risk_score', ascending=False).head(10)
    for _, row in top10.iterrows():
        print(f"TX {row['TransactionID']} (Risk: {row['risk_score']:.2f}, Fraud={row['isFraud']})")
        print(f"  Reasons: {row['top_reasons']}")
        print(f"  Evidence: {row['evidence_json']}\n")

if __name__ == "__main__":
    print("Loading Graph from Phase 0...")
    raw_df = phase0.load_data()
    G = phase0.build_graph(raw_df)
    
    scorer = FraudScorer(G)
    df_scores = scorer.run()
    
    df_scores.to_csv(OUTPUT_FILE, index=False)
    print(f"Scores saved to {OUTPUT_FILE}")
    
    evaluate(df_scores)
