# 📊 Customer Segmentation & Hybrid Recommendation System

**RFM Analysis · KMeans Clustering · PCA · Cosine Similarity-based Recommendations**

🔗 **Live Demo:** [customer-analytics-app-hwzxlwweknwpstpgkmevrl.streamlit.app](https://customer-analytics-app-hwzxlwweknwpstpgkmevrl.streamlit.app/)

---

## 🧭 Overview

An end-to-end customer analytics dashboard built on the **UCI Online Retail** dataset. It segments customers using **RFM (Recency, Frequency, Monetary) analysis** and **KMeans clustering (k=4)**, then powers a **hybrid recommendation engine** that combines cluster-level purchase trends with **cosine-similarity** product relationships.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-Learn (KMeans, PCA, PowerTransformer, StandardScaler) |
| Recommender | Cosine Similarity (product–customer matrix) |
| Visualisation | Plotly |
| Dashboard | Streamlit |

---

## ⚙️ ML Pipeline

1. **Data Cleaning** — remove cancelled orders, negative quantities, missing CustomerIDs
2. **Feature Engineering** — `TotalAmount = Quantity × UnitPrice`
3. **RFM Analysis** — Recency, Frequency, Monetary per customer
4. **Yeo-Johnson Transformation** — reduce skewness in RFM features
5. **StandardScaler** — normalise features for clustering
6. **KMeans Clustering (k=4)** — segment customers into 4 behavioural groups
7. **PCA** — 2D visualisation of cluster separation
8. **Cosine Similarity** — binary product–purchase matrix for product-to-product similarity
9. **Hybrid Recommender** — cluster top-products + similar products → deduplicated Top-10

---

## 🧩 Customer Segments (k=4)

| Cluster | Segment | RFM Profile |
|---|---|---|
| ⭐ Cluster 3 | **VIP Customers** | Low Recency · High Frequency · High Monetary |
| 🔵 Cluster 0 | **Regular Customers** | Medium across all RFM metrics |
| 🟡 Cluster 2 | **Low Engagement** | Recent but infrequent, low spend |
| 🔴 Cluster 1 | **Lost Customers** | High Recency · Very low activity |

---

## 📱 Dashboard Pages

- 🏠 **Home** — project overview, tech badges, KPIs, pipeline summary
- 📋 **Data Overview** — dataset shape, missing values, preview, summary stats, RFM table
- 🧩 **Customer Segmentation** — cluster distribution, RFM averages, PCA scatter, heatmap
- 🎯 **Recommendation System** — search by Customer ID → profile, purchase history, hybrid recommendations
- 📈 **Visualisation Dashboard** — interactive Plotly charts (revenue, frequency, recency, top products)
- 💡 **Project Insights** — segment-wise business recommendations

---

## 🚀 Run Locally

```bash
git clone https://github.com/soham259/customer-analytics-app.git
cd customer-analytics-app
pip install -r requirements.txt
streamlit run app.py
```

The app loads pre-computed model artefacts from `data/` and `models/`. To regenerate them from the raw dataset:

```bash
python build_artifacts.py
```

---

## 📁 Project Structure

```
customer-analytics-app/
├── app.py                  # Main Streamlit app (6 pages)
├── utils.py                # Data loaders, caching, helpers
├── recommendation.py       # Hybrid recommender logic
├── plots.py                 # Plotly chart functions
├── build_artifacts.py      # Rebuilds pkl artefacts from raw data
├── requirements.txt
├── data/                    # Dataset + processed pkl files
└── models/                  # Trained KMeans, PCA, Scaler, Transformer
```

---

## 📈 Dataset

[UCI Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail) — transactional data from a UK-based online retailer (Dec 2010 – Dec 2011).
