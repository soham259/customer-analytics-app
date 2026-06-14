"""
utils.py  –  Data loading, caching, and shared helper functions.
"""

import pandas as pd
import numpy as np
import joblib
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR   = "data"
MODELS_DIR = "models"

# Human-readable cluster names derived from RFM means (k=4, business labels)
CLUSTER_LABELS = {
    0: "Regular Customers",
    1: "Lost Customers",
    2: "Low Engagement",
    3: "VIP Customers",
}

CLUSTER_COLORS = {
    0: "#3B82F6",   # blue
    1: "#EF4444",   # red
    2: "#F59E0B",   # amber
    3: "#10B981",   # emerald
}

CLUSTER_ICONS = {
    0: "🔵",
    1: "🔴",
    2: "🟡",
    3: "🟢",
}

# ─────────────────────────────────────────────────────────────────────────────
# CACHED LOADERS  (run once per session)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_rfm() -> pd.DataFrame:
    """Return RFM DataFrame with kmeans_cluster column."""
    return pd.read_pickle(f"{DATA_DIR}/rfm_4.pkl")


@st.cache_data(show_spinner=False)
def load_df_rev() -> pd.DataFrame:
    """Return the cleaned transaction DataFrame."""
    return pd.read_pickle(f"{DATA_DIR}/df_rev.pkl")


@st.cache_data(show_spinner=False)
def load_df_merged() -> pd.DataFrame:
    """Return transactions merged with cluster labels."""
    return pd.read_pickle(f"{DATA_DIR}/df_merged.pkl")


@st.cache_data(show_spinner=False)
def load_cluster_products() -> pd.DataFrame:
    """Return cluster-product sorted DataFrame."""
    return pd.read_pickle(f"{DATA_DIR}/cluster_product_sorted.pkl")


@st.cache_data(show_spinner=False)
def load_product_similarity() -> pd.DataFrame:
    """Return cosine-similarity DataFrame (products × products)."""
    return pd.read_pickle(f"{DATA_DIR}/product_similarity_df.pkl")


@st.cache_data(show_spinner=False)
def load_pca_df() -> pd.DataFrame:
    """Return PCA 2-component DataFrame with Cluster column."""
    return pd.read_pickle(f"{DATA_DIR}/pca_df.pkl")


@st.cache_resource(show_spinner=False)
def load_models() -> dict:
    """Load and return all sklearn model objects."""
    return {
        "kmeans":            joblib.load(f"{MODELS_DIR}/kmeans_k4.pkl"),
        "power_transformer": joblib.load(f"{MODELS_DIR}/power_transformer.pkl"),
        "scaler":            joblib.load(f"{MODELS_DIR}/standard_scaler.pkl"),
        "pca":               joblib.load(f"{MODELS_DIR}/pca.pkl"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def cluster_label(cluster_id: int) -> str:
    return CLUSTER_LABELS.get(int(cluster_id), f"Cluster {cluster_id}")


def cluster_color(cluster_id: int) -> str:
    return CLUSTER_COLORS.get(int(cluster_id), "#6B7280")


def cluster_icon(cluster_id: int) -> str:
    return CLUSTER_ICONS.get(int(cluster_id), "⚪")


def get_valid_customer_ids() -> list:
    rfm = load_rfm()
    return sorted(rfm.index.tolist())


def is_valid_customer(customer_id: int) -> bool:
    return customer_id in load_rfm().index


def rfm_with_labels(rfm: pd.DataFrame) -> pd.DataFrame:
    """Attach a human-readable 'Segment' column to the RFM DataFrame."""
    df = rfm.copy()
    df["Segment"] = df["kmeans_cluster"].map(CLUSTER_LABELS)
    df["Color"]   = df["kmeans_cluster"].map(CLUSTER_COLORS)
    return df


def format_currency(value: float) -> str:
    if value >= 1_000_000:
        return f"£{value/1_000_000:.1f}M"
    if value >= 1_000:
        return f"£{value/1_000:.1f}K"
    return f"£{value:,.0f}"


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Return a styled summary statistics table."""
    return df.describe().round(2)
