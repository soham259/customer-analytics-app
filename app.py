"""
app.py  –  Customer Segmentation & Hybrid Recommendation System
           Streamlit Dashboard  |  KMeans k=4  |  RFM + Cosine Similarity
"""

import io
import streamlit as st
import pandas as pd
import numpy as np

# ── Internal modules ──────────────────────────────────────────────────────────
from utils import (
    load_rfm, load_df_rev, load_df_merged, load_pca_df,
    rfm_with_labels, format_currency, summary_statistics,
    CLUSTER_LABELS, CLUSTER_COLORS, CLUSTER_ICONS,
)
from recommendation import hybrid_recommend, get_customer_profile
from plots import (
    cluster_distribution_bar, cluster_distribution_pie,
    rfm_averages_bar, pca_scatter,
    monetary_by_cluster, frequency_by_cluster, recency_by_cluster,
    rfm_heatmap, top_products_bar, monthly_revenue,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Hide top loading/progress bar ── */
div[data-testid="stDecoration"] { display: none; }
div[data-testid="stStatusWidget"] { display: none; }

/* ── Base ── */
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] * { color: #CBD5E1 !important; }

/* Sidebar title */
section[data-testid="stSidebar"] h3 {
    font-size: 1.15rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.3px;
    padding: 0.4rem 0 0.2rem;
}

/* Radio nav items → card style */
section[data-testid="stSidebar"] .stRadio [role="radiogroup"] {
    gap: 0.35rem;
    display: flex;
    flex-direction: column;
}
section[data-testid="stSidebar"] .stRadio label {
    font-size: 0.92rem !important;
    font-weight: 500;
    padding: 0.55rem 0.8rem;
    border-radius: 10px;
    border: 1px solid transparent;
    transition: background 0.15s, border-color 0.15s;
    width: 100%;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(59,130,246,0.10);
    border-color: rgba(59,130,246,0.25);
}
section[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] input:checked + div {
    background: rgba(59,130,246,0.18) !important;
}
section[data-testid="stSidebar"] .stRadio label:has(input:checked) {
    background: rgba(59,130,246,0.16);
    border-color: rgba(59,130,246,0.4);
}
section[data-testid="stSidebar"] .stRadio label:has(input:checked) p {
    color: #93C5FD !important;
    font-weight: 700;
}
/* Hide the radio circle bullets for a cleaner nav look */
section[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] > div:first-child {
    display: none;
}
section[data-testid="stSidebar"] .stRadio label {
    cursor: pointer;
}

/* Sidebar divider */
section[data-testid="stSidebar"] hr {
    margin: 1.1rem 0;
    border-color: rgba(255,255,255,0.08);
}

/* Sidebar info box */
.sidebar-info {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 0.85rem 1rem;
    font-size: 0.8rem;
    line-height: 1.9;
    color: #94A3B8;
}
.sidebar-info b { color: #CBD5E1; }
.sidebar-info .dot {
    display:inline-block; width:7px; height:7px; border-radius:50%;
    margin-right:7px; vertical-align:middle;
}

/* ── Main background ── */
.main .block-container { padding: 3.5rem 2rem 1.5rem; max-width: 1280px; }
.stApp { background: #0F172A; color: #E2E8F0; }

/* ── Cards ── */
.card {
    background: #1E293B;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
    transition: box-shadow 0.2s;
}
.card:hover { box-shadow: 0 4px 20px rgba(59,130,246,0.18); }

/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background: #1E293B;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1rem;
}

/* ── Badges ── */
.badge {
    display: inline-block;
    background: rgba(59,130,246,0.18);
    color: #93C5FD;
    border: 1px solid rgba(59,130,246,0.3);
    border-radius: 999px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 3px 3px;
}
.badge-green  { background:rgba(16,185,129,.18); color:#6EE7B7; border-color:rgba(16,185,129,.35); }
.badge-purple { background:rgba(139,92,246,.18); color:#C4B5FD; border-color:rgba(139,92,246,.35); }
.badge-orange { background:rgba(245,158,11,.18); color:#FDE68A; border-color:rgba(245,158,11,.35); }

/* ── Section headers ── */
.section-header {
    font-size: 1.6rem;
    font-weight: 700;
    color: #F1F5F9;
    margin-bottom: 0.3rem;
}
.section-sub {
    font-size: 0.93rem;
    color: #94A3B8;
    margin-bottom: 1.4rem;
}

/* ── Rec cards ── */
.rec-card {
    background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
    border: 1px solid rgba(59,130,246,0.25);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.88rem;
    color: #CBD5E1;
}
.rec-card strong { color: #93C5FD; font-size:0.75rem; display:block; margin-bottom:2px; }

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.07); }

/* ── Expanders ── */
details summary { color: #93C5FD !important; font-weight: 600; }

/* ── Table ── */
.stDataFrame { background: #1E293B !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='display:flex; align-items:center; gap:10px; padding: 0.4rem 0 0.8rem;'>
        <div style='font-size:1.8rem;'>📊</div>
        <div>
            <div style='font-size:1.15rem; font-weight:800; color:#F1F5F9; line-height:1.2;'>Analytics Hub</div>
            <div style='font-size:0.72rem; color:#64748B; font-weight:500;'>Customer Intelligence Suite</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    PAGES = {
        "🏠  Home":                   "home",
        "📋  Data Overview":          "data",
        "🧩  Customer Segmentation":  "segmentation",
        "🎯  Recommendation System":  "recommendation",
        "📈  Visualisation Dashboard": "viz",
        "💡  Project Insights":        "insights",
    }

    st.markdown("<div style='font-size:0.72rem; font-weight:700; color:#64748B; "
                "letter-spacing:1px; text-transform:uppercase; margin-bottom:0.3rem;'>"
                "Navigation</div>", unsafe_allow_html=True)
    choice = st.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed")
    page   = PAGES[choice]

    st.markdown("---")
    st.markdown("<div style='font-size:0.72rem; font-weight:700; color:#64748B; "
                "letter-spacing:1px; text-transform:uppercase; margin-bottom:0.5rem;'>"
                "Pipeline Summary</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='sidebar-info'>
        <div><span class='dot' style='background:#3B82F6;'></span><b>Model</b> &nbsp;KMeans (k = 4)</div>
        <div><span class='dot' style='background:#10B981;'></span><b>Features</b> &nbsp;RFM + PCA</div>
        <div><span class='dot' style='background:#F59E0B;'></span><b>Recommender</b> &nbsp;Cosine Similarity</div>
        <div><span class='dot' style='background:#8B5CF6;'></span><b>Transform</b> &nbsp;Yeo-Johnson</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:0.68rem; color:#475569; text-align:center; "
                "margin-top:1.5rem;'>Customer Segmentation &amp; Hybrid Recommendation System</div>",
                unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA  (cached)
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading data…"):
    rfm       = load_rfm()
    df_rev    = load_df_rev()
    df_merged = load_df_merged()
    pca_df    = load_pca_df()

rfm_labeled = rfm_with_labels(rfm)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 – HOME
# ═════════════════════════════════════════════════════════════════════════════
if page == "home":
    # Hero
    st.markdown("""
    <div style='text-align:center; padding: 2rem 0 1rem;'>
        <div style='font-size:3.5rem;'>📊</div>
        <h1 style='font-size:2.2rem; font-weight:800; color:#F1F5F9; margin:0.4rem 0;'>
            Customer Segmentation &amp; Hybrid<br>Recommendation System
        </h1>
        <p style='color:#94A3B8; font-size:1.05rem; max-width:680px; margin:0.6rem auto;'>
            Using RFM Analysis and Machine Learning to identify customer segments
            and deliver personalised product recommendations.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Tech badges
    badges = [
        ("Python",           "badge"),
        ("Pandas",           "badge"),
        ("Scikit-Learn",     "badge-green"),
        ("Streamlit",        "badge-purple"),
        ("KMeans (k=4)",     "badge"),
        ("PCA",              "badge-orange"),
        ("Cosine Similarity","badge-green"),
        ("Yeo-Johnson",      "badge-orange"),
        ("XGBoost",          "badge-purple"),
    ]
    badge_html = "".join(f'<span class="badge {cls}">{label}</span>' for label, cls in badges)
    st.markdown(f"<div style='text-align:center; margin: 0.5rem 0 2rem;'>{badge_html}</div>",
                unsafe_allow_html=True)

    st.markdown("---")

    # KPI row
    total_customers  = rfm.shape[0]
    total_revenue    = rfm["Monetary"].sum()
    avg_order_value  = df_rev["TotalAmount"].mean()
    vip_count        = (rfm["kmeans_cluster"] == 3).sum()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("👥 Total Customers",  f"{total_customers:,}")
    k2.metric("💰 Total Revenue",    format_currency(total_revenue))
    k3.metric("🛒 Avg Order Value",  f"£{avg_order_value:.2f}")
    k4.metric("⭐ VIP Customers",    f"{vip_count:,}")

    st.markdown("---")

    # Pipeline overview
    st.markdown("<p class='section-header'>🔧 ML Pipeline</p>", unsafe_allow_html=True)

    steps = [
        ("1️⃣", "Data Cleaning",          "Remove cancelled orders, negative values, missing CustomerIDs"),
        ("2️⃣", "Feature Engineering",    "TotalAmount = Quantity × UnitPrice"),
        ("3️⃣", "RFM Analysis",           "Recency · Frequency · Monetary computed per customer"),
        ("4️⃣", "Yeo-Johnson Transform",  "Reduce skewness in RFM features"),
        ("5️⃣", "Standard Scaling",       "Zero mean, unit variance for clustering"),
        ("6️⃣", "KMeans Clustering",      "k=4 chosen for business interpretability"),
        ("7️⃣", "PCA Visualisation",      "2-component scatter to inspect cluster separation"),
        ("8️⃣", "Cosine Similarity",      "Product-level binary purchase matrix"),
        ("9️⃣", "Hybrid Recommender",     "Cluster products + similar products → deduplicated top-10"),
    ]

    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(steps):
        with cols[i % 3]:
            st.markdown(f"""
            <div class='card'>
                <div style='font-size:1.4rem;'>{icon}</div>
                <div style='font-weight:700; color:#F1F5F9; margin:4px 0;'>{title}</div>
                <div style='font-size:0.82rem; color:#94A3B8;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 – DATA OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
elif page == "data":
    st.markdown("<p class='section-header'>📋 Data Overview</p>", unsafe_allow_html=True)
    st.markdown("<p class='section-sub'>Online Retail dataset — after cleaning & deduplication</p>",
                unsafe_allow_html=True)

    # Shape + missing
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📦 Rows",            f"{df_rev.shape[0]:,}")
    c2.metric("📊 Columns",         f"{df_rev.shape[1]}")
    c3.metric("👤 Unique Customers", f"{df_rev['CustomerID'].nunique():,}")
    c4.metric("🛍️ Unique Products",  f"{df_rev['Description'].nunique():,}")

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["📄 Preview", "🔍 Missing Values",
                                       "📊 Statistics", "🗺️ RFM Table"])

    with tab1:
        n = st.slider("Rows to preview", 5, 100, 20)
        st.dataframe(df_rev.head(n), use_container_width=True, height=400)

    with tab2:
        missing = df_rev.isnull().sum().reset_index()
        missing.columns = ["Column", "Missing Count"]
        missing["Missing %"] = (missing["Missing Count"] / df_rev.shape[0] * 100).round(2)
        missing = missing.sort_values("Missing Count", ascending=False)
        st.dataframe(missing, use_container_width=True)
        st.success("✅ After cleaning: 0 missing CustomerIDs")

    with tab3:
        st.markdown("##### Numerical Summary")
        st.dataframe(summary_statistics(df_rev.select_dtypes(include="number")),
                     use_container_width=True)

    with tab4:
        show_rfm = rfm_labeled[["Recency","Frequency","Monetary","Segment"]].copy()
        show_rfm.index.name = "CustomerID"
        st.dataframe(show_rfm.reset_index(), use_container_width=True, height=420)

        # Download
        csv_buf = io.StringIO()
        show_rfm.reset_index().to_csv(csv_buf, index=False)
        st.download_button("⬇️ Download RFM CSV", csv_buf.getvalue(),
                           "rfm_segments.csv", "text/csv")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 – CUSTOMER SEGMENTATION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "segmentation":
    st.markdown("<p class='section-header'>🧩 Customer Segmentation</p>", unsafe_allow_html=True)
    st.markdown("<p class='section-sub'>KMeans k=4 · Yeo-Johnson · StandardScaler · PCA</p>",
                unsafe_allow_html=True)

    # Cluster summary cards
    st.markdown("#### Cluster Summary")
    seg_cols = st.columns(4)
    for cluster_id, label in CLUSTER_LABELS.items():
        count = (rfm["kmeans_cluster"] == cluster_id).sum()
        icon  = CLUSTER_ICONS[cluster_id]
        color = CLUSTER_COLORS[cluster_id]
        pct   = count / len(rfm) * 100
        with seg_cols[cluster_id]:
            st.markdown(f"""
            <div class='card' style='border-top: 3px solid {color};'>
                <div style='font-size:1.8rem;'>{icon}</div>
                <div style='font-weight:700; color:{color}; font-size:1rem;'>{label}</div>
                <div style='font-size:1.8rem; font-weight:800; color:#F1F5F9; margin:4px 0;'>{count:,}</div>
                <div style='font-size:0.82rem; color:#94A3B8;'>{pct:.1f}% of customers</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distribution", "📉 RFM Averages",
                                       "🔵 PCA Scatter",  "📋 Cluster Table"])

    with tab1:
        ca, cb = st.columns(2)
        with ca:
            st.plotly_chart(cluster_distribution_bar(rfm), use_container_width=True)
        with cb:
            st.plotly_chart(cluster_distribution_pie(rfm), use_container_width=True)

    with tab2:
        st.plotly_chart(rfm_averages_bar(rfm), use_container_width=True)
        st.plotly_chart(rfm_heatmap(rfm), use_container_width=True)

    with tab3:
        st.plotly_chart(pca_scatter(pca_df), use_container_width=True)
        with st.expander("ℹ️ About PCA"):
            st.write("""
            **Principal Component Analysis (PCA)** reduces the 3 RFM dimensions
            to 2 principal components (PC1, PC2) for visual inspection.
            Each dot represents one customer. Well-separated clusters confirm
            that KMeans with k=4 found meaningful structure in the data.
            """)

    with tab4:
        means = (rfm.groupby("kmeans_cluster")[["Recency","Frequency","Monetary"]]
                    .agg(["mean","min","max"])
                    .round(1))
        means.index = [f"{CLUSTER_ICONS[k]} {CLUSTER_LABELS[k]}" for k in means.index]
        st.dataframe(means, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 – RECOMMENDATION SYSTEM
# ═════════════════════════════════════════════════════════════════════════════
elif page == "recommendation":
    st.markdown("<p class='section-header'>🎯 Hybrid Recommendation System</p>",
                unsafe_allow_html=True)
    st.markdown("""
    <p class='section-sub'>
        Cluster-based top products + Cosine Similarity product pairs → Top-10 personalised recs
    </p>""", unsafe_allow_html=True)

    # ── Input form ────────────────────────────────────────────────────────────
    with st.form("rec_form"):
        inp_col, btn_col = st.columns([4, 1])
        with inp_col:
            cid_input = st.text_input("🔍 Enter Customer ID",
                                      placeholder="e.g. 17850",
                                      label_visibility="collapsed")
        with btn_col:
            submitted = st.form_submit_button("Get Recommendations", use_container_width=True)

    # Quick-pick sample IDs
    sample_ids = [17850, 15311, 13408, 16503, 14911]
    st.caption("Sample IDs: " + "  ·  ".join(str(i) for i in sample_ids))

    if submitted and cid_input.strip():
        try:
            customer_id = int(cid_input.strip())
        except ValueError:
            st.error("❌ Please enter a valid numeric Customer ID.")
            st.stop()

        with st.spinner("🔄 Building personalised recommendations…"):
            result = hybrid_recommend(customer_id)

        if result.get("error"):
            st.error(f"❌ {result['error']}")
            st.info(f"💡 Valid IDs range from {rfm.index.min()} to {rfm.index.max()}. "
                    f"Try: {sample_ids[0]}")
        else:
            profile = get_customer_profile(customer_id)

            # ── Customer profile ───────────────────────────────────────────────
            st.markdown("---")
            st.markdown("#### 👤 Customer Profile")
            p1, p2, p3, p4, p5 = st.columns(5)
            p1.metric("Customer ID",  str(customer_id))
            p2.metric("📅 Recency",   f"{profile['Recency']} days")
            p3.metric("🔁 Frequency", f"{profile['Frequency']} orders")
            p4.metric("💰 Monetary",  f"£{profile['Monetary']:,.2f}")
            p5.metric("🏷️ Segment",
                      f"{profile['cluster_icon']} {profile['cluster_label']}")

            clr = CLUSTER_COLORS[result["cluster_id"]]
            st.markdown(f"""
            <div class='card' style='border-left: 4px solid {clr}; margin-top:0.5rem;'>
                <span style='font-size:1.5rem;'>{result['cluster_icon']}</span>
                <span style='color:{clr}; font-weight:700; font-size:1rem; margin-left:6px;'>
                    {result['cluster_label']}
                </span>
            </div>
            """, unsafe_allow_html=True)

            # ── Tabs ──────────────────────────────────────────────────────────
            rt1, rt2, rt3 = st.tabs(["🎯 Final Recommendations",
                                      "🛒 Purchase History",
                                      "🔍 Rec Breakdown"])

            with rt1:
                st.success(f"✅ {len(result['final_recs'])} personalised recommendations generated!")
                if result["final_recs"]:
                    rec_cols = st.columns(2)
                    for idx, prod in enumerate(result["final_recs"]):
                        with rec_cols[idx % 2]:
                            st.markdown(f"""
                            <div class='rec-card'>
                                <strong>#{idx+1}</strong>{prod}
                            </div>
                            """, unsafe_allow_html=True)

                    # Download
                    rec_df  = pd.DataFrame({"Rank": range(1, len(result["final_recs"]) + 1),
                                            "Product": result["final_recs"]})
                    csv_out = rec_df.to_csv(index=False)
                    st.download_button("⬇️ Download Recommendations CSV",
                                       csv_out,
                                       f"recommendations_{customer_id}.csv",
                                       "text/csv")
                else:
                    st.warning("No recommendations could be generated.")

            with rt2:
                hist = result["purchase_history"]
                st.markdown(f"**{len(hist)} products purchased** by Customer {customer_id}:")
                hist_df = pd.DataFrame({"Product": hist})
                hist_df.index += 1
                st.dataframe(hist_df, use_container_width=True, height=320)

            with rt3:
                ca, cb = st.columns(2)
                with ca:
                    st.markdown(f"**🧩 Cluster-Based Recs** ({len(result['cluster_recs'])})")
                    for i, p in enumerate(result["cluster_recs"], 1):
                        st.markdown(f"""
                        <div class='rec-card'><strong>#{i}</strong>{p}</div>
                        """, unsafe_allow_html=True)
                with cb:
                    st.markdown(f"**🔗 Similarity-Based Recs** ({len(result['similar_recs'])})")
                    for i, p in enumerate(result["similar_recs"][:10], 1):
                        st.markdown(f"""
                        <div class='rec-card'><strong>#{i}</strong>{p}</div>
                        """, unsafe_allow_html=True)

    elif submitted:
        st.warning("⚠️ Please enter a Customer ID.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 – VISUALISATION DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
elif page == "viz":
    st.markdown("<p class='section-header'>📈 Visualisation Dashboard</p>",
                unsafe_allow_html=True)
    st.markdown("<p class='section-sub'>Interactive Plotly charts – explore every angle</p>",
                unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["💰 Revenue", "🔁 Frequency",
                                       "📅 Recency",  "🛍️ Products"])

    with tab1:
        st.plotly_chart(monetary_by_cluster(rfm),   use_container_width=True)
        st.plotly_chart(monthly_revenue(df_rev),    use_container_width=True)

    with tab2:
        st.plotly_chart(frequency_by_cluster(rfm),  use_container_width=True)
        st.plotly_chart(cluster_distribution_bar(rfm), use_container_width=True)

    with tab3:
        st.plotly_chart(recency_by_cluster(rfm),    use_container_width=True)
        st.plotly_chart(pca_scatter(pca_df),        use_container_width=True)

    with tab4:
        n_top = st.slider("Top N products", 5, 30, 10)
        st.plotly_chart(top_products_bar(df_merged, top_n=n_top), use_container_width=True)

        with st.expander("🗂️ Product List"):
            top_prods = (df_merged.groupby("Description")["Quantity"]
                                  .sum()
                                  .nlargest(n_top)
                                  .reset_index()
                                  .rename(columns={"Quantity": "Total Qty"}))
            top_prods.index += 1
            st.dataframe(top_prods, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 6 – PROJECT INSIGHTS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "insights":
    st.markdown("<p class='section-header'>💡 Project Insights</p>", unsafe_allow_html=True)
    st.markdown("<p class='section-sub'>Business intelligence derived from RFM clustering</p>",
                unsafe_allow_html=True)

    # Segment insights
    insight_data = {
        3: {
            "icon":  "⭐",
            "color": "#10B981",
            "title": "VIP Customers  (Cluster 3)",
            "rfm":   "Low Recency · High Frequency · High Monetary",
            "desc":  """
            These are your most valuable customers. They purchase frequently, 
            spend the most, and bought very recently. Even though they represent 
            a small share of the base, they contribute disproportionately large revenue.
            """,
            "actions": [
                "Launch an exclusive loyalty / VIP rewards programme.",
                "Offer early access to new products and seasonal collections.",
                "Assign a dedicated account manager for top spenders.",
                "Run personalised upsell campaigns based on purchase history.",
            ],
        },
        0: {
            "icon":  "🔵",
            "color": "#3B82F6",
            "title": "Regular Customers  (Cluster 0)",
            "rfm":   "Medium Recency · Low-Medium Frequency · Medium Monetary",
            "desc":  """
            The backbone of revenue. These customers purchase at a reasonable cadence 
            and have moderate spending. With the right nudges they can be elevated 
            into the VIP segment.
            """,
            "actions": [
                "Run targeted cross-sell campaigns using the recommendation engine.",
                "Offer volume discounts to increase basket size.",
                "Send re-engagement emails when recency begins to rise.",
                "Feature curated product bundles to encourage repeat visits.",
            ],
        },
        2: {
            "icon":  "🟡",
            "color": "#F59E0B",
            "title": "Low Engagement  (Cluster 2)",
            "rfm":   "Low Recency · Very Low Frequency · Low Monetary",
            "desc":  """
            Recent but infrequent buyers. They discovered the store recently but 
            haven't built a habit. Converting them to Regular customers is the 
            highest-leverage growth opportunity.
            """,
            "actions": [
                "Trigger onboarding email sequences with popular product highlights.",
                "Offer first-repeat-purchase discount (e.g. 10% off second order).",
                "Use personalised homepage banners based on browsing/purchase history.",
                "A/B test free-shipping thresholds to encourage higher spend.",
            ],
        },
        1: {
            "icon":  "🔴",
            "color": "#EF4444",
            "title": "Lost Customers  (Cluster 1)",
            "rfm":   "High Recency · Very Low Frequency · Low Monetary",
            "desc":  """
            Customers who haven't purchased in a long time and were already infrequent 
            buyers. The priority is win-back; if they don't respond, focus spend on 
            the other three segments.
            """,
            "actions": [
                "Send automated win-back campaign: 'We miss you – here's 15% off'.",
                "Survey churned customers to understand exit reasons.",
                "Retarget with paid ads featuring best-sellers they haven't seen.",
                "If no response after 3 touches, suppress to reduce marketing spend.",
            ],
        },
    }

    for cid, info in insight_data.items():
        count = (rfm["kmeans_cluster"] == cid).sum()
        monetary = rfm[rfm["kmeans_cluster"] == cid]["Monetary"].sum()
        pct = count / len(rfm) * 100

        with st.expander(f"{info['icon']}  {info['title']}  —  {count:,} customers ({pct:.1f}%)",
                         expanded=(cid == 3)):
            ca, cb = st.columns([2, 1])
            with ca:
                st.markdown(f"""
                <div style='
                    border-left: 3px solid {info["color"]};
                    padding-left: 1rem;
                    color: #CBD5E1;
                    font-size: 0.95rem;
                    line-height: 1.7;
                '>
                    <b style='color:{info["color"]};'>RFM Profile:</b> {info["rfm"]}<br><br>
                    {info["desc"].strip()}
                </div>
                """, unsafe_allow_html=True)

                st.markdown("**📌 Recommended Actions:**")
                for action in info["actions"]:
                    st.markdown(f"• {action}")
            with cb:
                st.metric("👥 Customers",  f"{count:,}")
                st.metric("💰 Total Rev",  format_currency(monetary))
                st.metric("📊 Share",      f"{pct:.1f}%")
                rev_share = monetary / rfm["Monetary"].sum() * 100
                st.metric("💹 Rev Share",  f"{rev_share:.1f}%")

    st.markdown("---")

    # Overall summary
    st.markdown("#### 🏁 Overall Business Summary")
    st.markdown("""
    <div class='card'>
        <ul style='color:#CBD5E1; line-height:2; margin:0; padding-left:1.2rem;'>
            <li>The majority of customers are <b>Regular</b> – they form the stable revenue backbone.</li>
            <li><b>VIP customers</b> are few but contribute the highest revenue per customer; retention is critical.</li>
            <li>A significant portion are <b>Lost Customers</b>, signalling a need for stronger retention/win-back campaigns.</li>
            <li><b>Low Engagement</b> customers are a high-potential pool for conversion – they bought recently and just need a nudge.</li>
            <li>The hybrid recommender surfaces relevant products for every segment, reducing the effort needed for manual merchandising.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)