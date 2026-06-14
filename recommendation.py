"""
recommendation.py  –  Hybrid Recommendation System.

Logic (from notebook):
  1. Find the customer's KMeans cluster.
  2. Get the top cluster products (by total Quantity) that the customer
     has NOT already purchased.
  3. For every product in the customer's purchase history, find the top-N
     cosine-similar products.
  4. Combine both lists, deduplicate, exclude already-purchased items.
  5. Return the top 10 final recommendations.
"""

from __future__ import annotations
import pandas as pd
from utils import (
    load_rfm,
    load_df_merged,
    load_cluster_products,
    load_product_similarity,
    cluster_label,
    cluster_icon,
    is_valid_customer,
)


# ─────────────────────────────────────────────────────────────────────────────
# LOW-LEVEL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def recommend_by_cluster(cluster_id: int, top_n: int = 10) -> list[str]:
    """Return the top-N products bought by customers in *cluster_id*."""
    cps = load_cluster_products()
    products = cps[cps["Cluster"] == cluster_id]
    return products["Description"].head(top_n).tolist()


def recommend_similar_products(product_name: str, top_n: int = 5) -> list[str]:
    """Return the top-N cosine-similar products for *product_name*."""
    sim_df = load_product_similarity()
    if product_name not in sim_df.index:
        return []
    scores = sim_df[product_name].sort_values(ascending=False)
    return scores.iloc[1 : top_n + 1].index.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# HYBRID RECOMMENDER  (mirrors notebook `hybrid_recommend`)
# ─────────────────────────────────────────────────────────────────────────────

def hybrid_recommend(customer_id: int, top_n: int = 10) -> dict:
    """
    Run the hybrid recommendation pipeline for one customer.

    Returns
    -------
    dict with keys:
        customer_id   : int
        cluster_id    : int
        cluster_label : str
        cluster_icon  : str
        purchase_history : list[str]
        cluster_recs  : list[str]   – top cluster products (excl. history)
        similar_recs  : list[str]   – cosine-similar products (excl. history)
        final_recs    : list[str]   – combined, deduplicated, top-N
        error         : str | None
    """
    # ── Validate ──────────────────────────────────────────────────────────────
    if not is_valid_customer(customer_id):
        return {"error": f"Customer ID {customer_id} not found in the dataset."}

    rfm      = load_rfm()
    df_merged = load_df_merged()
    cps      = load_cluster_products()

    # ── 1. Customer cluster ───────────────────────────────────────────────────
    cluster_id = int(rfm.loc[customer_id, "kmeans_cluster"])

    # ── 2. Purchase history ───────────────────────────────────────────────────
    history = df_merged[df_merged["CustomerID"] == customer_id][
        "Description"
    ].unique().tolist()

    # ── 3. Cluster-based recs ─────────────────────────────────────────────────
    cluster_pool = cps[cps["Cluster"] == cluster_id]["Description"].head(20).tolist()
    cluster_recs = [p for p in cluster_pool if p not in history][:top_n]

    # ── 4. Similarity-based recs ──────────────────────────────────────────────
    similar_raw: list[str] = []
    for product in history:
        similar_raw.extend(recommend_similar_products(product, top_n=3))

    # deduplicate while preserving order
    seen: set[str] = set()
    similar_dedup: list[str] = []
    for p in similar_raw:
        if p not in seen:
            seen.add(p)
            similar_dedup.append(p)

    similar_recs = [p for p in similar_dedup if p not in history]

    # ── 5. Hybrid merge ───────────────────────────────────────────────────────
    combined_seen: set[str] = set()
    final_recs: list[str] = []
    for p in cluster_recs + similar_recs:
        if p not in combined_seen and p not in history:
            combined_seen.add(p)
            final_recs.append(p)
        if len(final_recs) >= top_n:
            break

    return {
        "error":           None,
        "customer_id":     customer_id,
        "cluster_id":      cluster_id,
        "cluster_label":   cluster_label(cluster_id),
        "cluster_icon":    cluster_icon(cluster_id),
        "purchase_history": list(history),
        "cluster_recs":    cluster_recs,
        "similar_recs":    similar_recs[:top_n],
        "final_recs":      final_recs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOMER PROFILE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_customer_profile(customer_id: int) -> dict:
    """Return RFM values + cluster for one customer."""
    rfm = load_rfm()
    if customer_id not in rfm.index:
        return {}
    row = rfm.loc[customer_id]
    return {
        "CustomerID":     customer_id,
        "Recency":        int(row["Recency"]),
        "Frequency":      int(row["Frequency"]),
        "Monetary":       float(row["Monetary"]),
        "cluster_id":     int(row["kmeans_cluster"]),
        "cluster_label":  cluster_label(int(row["kmeans_cluster"])),
        "cluster_icon":   cluster_icon(int(row["kmeans_cluster"])),
    }
