"""
build_artifacts.py

Run this ONCE locally (inside the customer_analytics_app folder) to
regenerate all .pkl files using YOUR installed numpy/pandas/sklearn
versions. This fixes "ModuleNotFoundError: No module named 'numpy._core...'"
errors caused by pkl files built with a different numpy version.

Usage:
    python build_artifacts.py
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import warnings

warnings.filterwarnings("ignore")

print("Loading data...")
df = pd.read_excel("data/Online_Retail.xlsx")
print("Raw shape:", df.shape)

# ── Clean ─────────────────────────────────────────────────────────────────
df_rev = df[(df["Quantity"] > 0) | (df["UnitPrice"] > 0)]
df_rev = df_rev[~df_rev["InvoiceNo"].astype(str).str.startswith("C")]
df_rev = df_rev.dropna(subset=["CustomerID"])
df_rev = df_rev.drop_duplicates()
df_rev["CustomerID"] = df_rev["CustomerID"].astype(int)
df_rev["TotalAmount"] = df_rev["Quantity"] * df_rev["UnitPrice"]
print("Clean shape:", df_rev.shape)

# ── RFM ───────────────────────────────────────────────────────────────────
reference_date = df_rev["InvoiceDate"].max() + pd.Timedelta(days=1)
rfm = df_rev.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (reference_date - x.max()).days,
    "InvoiceNo": "nunique",
    "TotalAmount": "sum"
})
rfm.columns = ["Recency", "Frequency", "Monetary"]
print("RFM shape:", rfm.shape)

# ── Yeo-Johnson ───────────────────────────────────────────────────────────
pt = PowerTransformer(method="yeo-johnson")
rfm_yeojohnson = pt.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])
rfm_yeo = pd.DataFrame(rfm_yeojohnson, columns=["Recency", "Frequency", "Monetary"], index=rfm.index)

# ── Standard Scaler ──────────────────────────────────────────────────────
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_yeo)
rfm_scaled = pd.DataFrame(rfm_scaled, columns=["Recency", "Frequency", "Monetary"], index=rfm.index)

# ── KMeans k=4 ────────────────────────────────────────────────────────────
rfm_4 = rfm[["Recency", "Frequency", "Monetary"]].copy()
kmeans_ = KMeans(n_clusters=4, random_state=42, n_init=20)
rfm_4["kmeans_cluster"] = kmeans_.fit_predict(rfm_scaled)
print("Cluster counts:", rfm_4["kmeans_cluster"].value_counts().sort_index().to_dict())

# ── PCA ───────────────────────────────────────────────────────────────────
pca = PCA(n_components=2, random_state=42)
rfm_pca = pca.fit_transform(rfm_scaled)
pca_df = pd.DataFrame(rfm_pca, columns=["PC1", "PC2"], index=rfm.index)
pca_df["Cluster"] = rfm_4["kmeans_cluster"]

# ── Merge clusters onto transactions ────────────────────────────────────
df_merged = df_rev.merge(rfm_4[["kmeans_cluster"]], left_on="CustomerID", right_index=True, how="inner")
df_merged.rename(columns={"kmeans_cluster": "Cluster"}, inplace=True)

# ── Cluster-Product ranking ─────────────────────────────────────────────
cluster_product = df_merged.groupby(["Cluster", "Description"])["Quantity"].sum().reset_index()
cluster_product_sorted = cluster_product.sort_values(["Cluster", "Quantity"], ascending=[True, False])

# ── Cosine Similarity ───────────────────────────────────────────────────
product_matrix = df_merged.pivot_table(index="Description", columns="CustomerID", values="Quantity", aggfunc="sum", fill_value=0)
product_matrix_binary = (product_matrix > 0).astype(int)
similarity = cosine_similarity(product_matrix_binary)
product_similarity_df = pd.DataFrame(similarity, index=product_matrix_binary.index, columns=product_matrix_binary.index)
print("Product matrix:", product_matrix_binary.shape)

# ── Save artefacts ───────────────────────────────────────────────────────
import os
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

joblib.dump(kmeans_, "models/kmeans_k4.pkl")
joblib.dump(pt, "models/power_transformer.pkl")
joblib.dump(scaler, "models/standard_scaler.pkl")
joblib.dump(pca, "models/pca.pkl")

rfm_4.to_pickle("data/rfm_4.pkl")
pca_df.to_pickle("data/pca_df.pkl")
df_merged.to_pickle("data/df_merged.pkl")
cluster_product_sorted.to_pickle("data/cluster_product_sorted.pkl")
product_similarity_df.to_pickle("data/product_similarity_df.pkl")
df_rev.to_pickle("data/df_rev.pkl")

print("\nAll artefacts rebuilt successfully ✓")
print("Now run: streamlit run app.py")
