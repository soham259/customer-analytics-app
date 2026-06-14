"""
plots.py  –  All Plotly visualisation functions for the Streamlit dashboard.
"""

from __future__ import annotations
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import CLUSTER_LABELS, CLUSTER_COLORS


# ── Shared theme ──────────────────────────────────────────────────────────────
_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", size=13, color="#E2E8F0"),
    margin=dict(l=10, r=10, t=50, b=10),
    hoverlabel=dict(bgcolor="#1E293B", font_color="#F1F5F9", font_size=13),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.1)",
                borderwidth=1),
)

_COLORS = list(CLUSTER_COLORS.values())   # [blue, red, amber, emerald]
_LABELS = list(CLUSTER_LABELS.values())


def _color_map(rfm: pd.DataFrame) -> dict[int, str]:
    return {k: CLUSTER_COLORS[k] for k in rfm["kmeans_cluster"].unique() if k in CLUSTER_COLORS}


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 – SEGMENTATION CHARTS
# ─────────────────────────────────────────────────────────────────────────────

def cluster_distribution_bar(rfm: pd.DataFrame) -> go.Figure:
    """Horizontal bar – customer count per cluster."""
    counts = (rfm["kmeans_cluster"]
              .map(CLUSTER_LABELS)
              .value_counts()
              .reset_index()
              .rename(columns={"index": "Segment", "kmeans_cluster": "Count"}))
    counts.columns = ["Segment", "Count"]

    color_seq = [CLUSTER_COLORS[k] for k, v in CLUSTER_LABELS.items()]
    fig = px.bar(
        counts, y="Segment", x="Count", orientation="h",
        color="Segment",
        color_discrete_sequence=color_seq,
        text="Count",
        title="Customer Count per Segment",
    )
    fig.update_traces(textposition="outside", textfont_size=13)
    fig.update_layout(**_LAYOUT, showlegend=False,
                      xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
                      yaxis=dict(showgrid=False))
    return fig


def cluster_distribution_pie(rfm: pd.DataFrame) -> go.Figure:
    """Donut chart – % customers per segment."""
    counts = rfm["kmeans_cluster"].value_counts().reset_index()
    counts.columns = ["Cluster", "Count"]
    counts["Segment"] = counts["Cluster"].map(CLUSTER_LABELS)
    counts["Color"]   = counts["Cluster"].map(CLUSTER_COLORS)

    fig = go.Figure(go.Pie(
        labels=counts["Segment"],
        values=counts["Count"],
        marker_colors=counts["Color"].tolist(),
        hole=0.45,
        textinfo="percent+label",
        textfont_size=13,
    ))
    fig.update_layout(**_LAYOUT, title="Segment Share (%)")
    return fig


def rfm_averages_bar(rfm: pd.DataFrame) -> go.Figure:
    """Grouped bar – RFM averages per cluster."""
    means = (rfm.groupby("kmeans_cluster")[["Recency", "Frequency", "Monetary"]]
               .mean()
               .reset_index())
    means["Segment"] = means["kmeans_cluster"].map(CLUSTER_LABELS)

    metrics = ["Recency", "Frequency", "Monetary"]
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=[f"Avg {m}" for m in metrics])

    for i, metric in enumerate(metrics, start=1):
        fig.add_trace(
            go.Bar(
                x=means["Segment"],
                y=means[metric],
                marker_color=[CLUSTER_COLORS[k] for k in means["kmeans_cluster"]],
                name=metric,
                showlegend=False,
                text=means[metric].round(1),
                textposition="outside",
            ),
            row=1, col=i,
        )

    fig.update_layout(**_LAYOUT, title="Average RFM Values by Segment",
                      height=380)
    fig.update_xaxes(tickangle=-20)
    return fig


def pca_scatter(pca_df: pd.DataFrame) -> go.Figure:
    """PCA scatter coloured by cluster."""
    df = pca_df.copy()
    df["Segment"] = df["Cluster"].map(CLUSTER_LABELS)
    df["Color"]   = df["Cluster"].map(CLUSTER_COLORS)
    df.index.name = "CustomerID"
    df = df.reset_index()

    fig = px.scatter(
        df, x="PC1", y="PC2",
        color="Segment",
        color_discrete_map={v: CLUSTER_COLORS[k] for k, v in CLUSTER_LABELS.items()},
        hover_data={"CustomerID": True, "PC1": ":.2f", "PC2": ":.2f"},
        title="PCA – Customer Segments (k=4)",
        opacity=0.7,
    )
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(
        **_LAYOUT, height=460,
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 – VISUALISATION DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

def monetary_by_cluster(rfm: pd.DataFrame) -> go.Figure:
    """Total monetary value per segment."""
    agg = (rfm.groupby("kmeans_cluster")["Monetary"]
              .sum()
              .reset_index())
    agg["Segment"] = agg["kmeans_cluster"].map(CLUSTER_LABELS)
    agg["Color"]   = agg["kmeans_cluster"].map(CLUSTER_COLORS)

    fig = go.Figure(go.Bar(
        x=agg["Segment"],
        y=agg["Monetary"],
        marker_color=agg["Color"].tolist(),
        text=agg["Monetary"].apply(lambda v: f"£{v:,.0f}"),
        textposition="outside",
    ))
    fig.update_layout(**_LAYOUT, title="Total Revenue (£) by Segment",
                      yaxis_tickprefix="£", yaxis_tickformat=",",
                      yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
                      xaxis=dict(showgrid=False))
    return fig


def frequency_by_cluster(rfm: pd.DataFrame) -> go.Figure:
    """Box plot of purchase frequency per segment."""
    df = rfm.copy()
    df["Segment"] = df["kmeans_cluster"].map(CLUSTER_LABELS)

    fig = px.box(
        df, x="Segment", y="Frequency",
        color="Segment",
        color_discrete_map={v: CLUSTER_COLORS[k] for k, v in CLUSTER_LABELS.items()},
        title="Purchase Frequency Distribution by Segment",
        points="outliers",
    )
    fig.update_layout(**_LAYOUT, showlegend=False,
                      yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"))
    return fig


def recency_by_cluster(rfm: pd.DataFrame) -> go.Figure:
    """Box plot of recency per segment."""
    df = rfm.copy()
    df["Segment"] = df["kmeans_cluster"].map(CLUSTER_LABELS)

    fig = px.box(
        df, x="Segment", y="Recency",
        color="Segment",
        color_discrete_map={v: CLUSTER_COLORS[k] for k, v in CLUSTER_LABELS.items()},
        title="Recency Distribution by Segment (lower = more recent)",
        points="outliers",
    )
    fig.update_layout(**_LAYOUT, showlegend=False,
                      yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"))
    return fig


def rfm_heatmap(rfm: pd.DataFrame) -> go.Figure:
    """Heatmap of average normalised RFM per cluster."""
    means = (rfm.groupby("kmeans_cluster")[["Recency", "Frequency", "Monetary"]]
               .mean())
    norm = (means - means.min()) / (means.max() - means.min())

    # Build text labels BEFORE renaming the index
    text_vals = [[f"{means.loc[cid, col]:.1f}" for col in means.columns]
                  for cid in means.index]

    norm.index = [CLUSTER_LABELS[k] for k in norm.index]

    fig = go.Figure(go.Heatmap(
        z=norm.values,
        x=norm.columns.tolist(),
        y=norm.index.tolist(),
        colorscale="RdYlGn",
        reversescale=False,
        text=text_vals,
        texttemplate="%{text}",
        textfont={"size": 12},
        colorbar=dict(title="Normalised"),
    ))
    fig.update_layout(**_LAYOUT, title="RFM Heatmap (Normalised Averages)", height=300)
    return fig


def top_products_bar(df_merged: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Top-N products by total quantity sold."""
    top = (df_merged.groupby("Description")["Quantity"]
                    .sum()
                    .nlargest(top_n)
                    .reset_index())
    fig = px.bar(
        top[::-1], x="Quantity", y="Description", orientation="h",
        text="Quantity", title=f"Top {top_n} Products by Quantity Sold",
        color="Quantity", color_continuous_scale="Blues",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(**_LAYOUT, showlegend=False, coloraxis_showscale=False,
                      yaxis=dict(showgrid=False),
                      xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
                      height=420)
    return fig


def monthly_revenue(df_rev: pd.DataFrame) -> go.Figure:
    """Monthly revenue trend line chart."""
    df = df_rev.copy()
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    monthly = df.groupby("YearMonth")["TotalAmount"].sum().reset_index()

    fig = px.line(
        monthly, x="YearMonth", y="TotalAmount",
        markers=True,
        title="Monthly Revenue Trend",
        labels={"TotalAmount": "Revenue (£)", "YearMonth": "Month"},
    )
    fig.update_traces(line_color="#3B82F6", marker=dict(color="#10B981", size=7))
    fig.update_layout(
        **_LAYOUT,
        yaxis_tickprefix="£", yaxis_tickformat=",",
        xaxis=dict(tickangle=-30, showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
    )
    return fig
