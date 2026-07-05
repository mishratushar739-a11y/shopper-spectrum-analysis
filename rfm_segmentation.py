"""
rfm_segmentation.py
--------------------
Step 3 of the Shopper Spectrum pipeline: Customer Segmentation.

Builds RFM (Recency, Frequency, Monetary) features for every customer,
scales them, finds a good number of clusters with the elbow method +
silhouette score, fits a KMeans model, and labels each cluster with a
business-friendly segment name based on its average RFM profile.

Artifacts saved to models/:
    - scaler.pkl              StandardScaler fit on log-transformed RFM
    - kmeans_model.pkl         Fitted KMeans model
    - cluster_segment_map.pkl  dict: cluster_id -> segment name
    - rfm_table.csv (data/)    Per-customer RFM + cluster + segment

Run directly (after data_cleaning.py):
    python src/rfm_segmentation.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
CLEAN_PATH = os.path.join(BASE_DIR, "data", "cleaned_retail.csv")
RFM_PATH = os.path.join(BASE_DIR, "data", "rfm_table.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
EDA_DIR = os.path.join(BASE_DIR, "outputs", "eda")

RANDOM_STATE = 42


def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
    ).reset_index()

    return rfm


def choose_k(X_scaled, k_range=range(2, 9)):
    inertias, sil_scores = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(list(k_range), inertias, marker="o")
    axes[0].set_title("Elbow Method")
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Inertia")

    axes[1].plot(list(k_range), sil_scores, marker="o", color="#C44E52")
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Score")

    fig.tight_layout()
    fig.savefig(os.path.join(EDA_DIR, "kmeans_k_selection.png"), dpi=120)
    plt.close(fig)

    best_k = list(k_range)[int(np.argmax(sil_scores))]
    print(f"Silhouette scores by k: {dict(zip(k_range, np.round(sil_scores, 3)))}")
    print(f"Best k by silhouette score: {best_k}")
    return best_k


def label_segments(rfm: pd.DataFrame, cluster_col="Cluster") -> dict:
    """Rank clusters by a composite RFM score and assign business names."""
    profile = rfm.groupby(cluster_col)[["Recency", "Frequency", "Monetary"]].mean()

    # Lower recency is better, higher frequency/monetary is better.
    # Build a composite rank score (average of normalized ranks).
    ranked = pd.DataFrame(index=profile.index)
    ranked["R_rank"] = profile["Recency"].rank(ascending=True)   # low recency -> good -> rank 1
    ranked["F_rank"] = profile["Frequency"].rank(ascending=False)  # high freq -> good -> rank 1
    ranked["M_rank"] = profile["Monetary"].rank(ascending=False)  # high monetary -> good -> rank 1
    ranked["score"] = ranked[["R_rank", "F_rank", "M_rank"]].mean(axis=1)
    ranked = ranked.sort_values("score")

    labels_in_order = ["High-Value Customers", "Regular Customers",
                        "Occasional Customers", "At-Risk Customers"]
    # if more/fewer clusters than 4 labels, extend/truncate gracefully
    n = len(ranked)
    if n <= 4:
        chosen_labels = labels_in_order[:n]
    else:
        chosen_labels = labels_in_order[:3] + \
            [f"Segment {i}" for i in range(1, n - 2)] + [labels_in_order[3]]

    cluster_to_label = {cluster_id: label for cluster_id, label in zip(ranked.index, chosen_labels)}
    return cluster_to_label


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(EDA_DIR, exist_ok=True)

    df = pd.read_csv(CLEAN_PATH, parse_dates=["InvoiceDate"])
    rfm = build_rfm(df)
    print(f"Built RFM table for {len(rfm):,} customers")
    print(rfm[["Recency", "Frequency", "Monetary"]].describe())

    # Log transform to reduce skew (add 1 to avoid log(0))
    rfm_log = rfm[["Recency", "Frequency", "Monetary"]].apply(lambda x: np.log1p(x))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm_log)

    # Elbow + silhouette plot is generated for reference, but we fix k=4
    # so clusters map cleanly onto the 4 business segments requested in
    # the project brief (High-Value / Regular / Occasional / At-Risk).
    choose_k(X_scaled)
    best_k = 4

    kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    rfm["Cluster"] = kmeans.fit_predict(X_scaled)

    cluster_to_label = label_segments(rfm)
    rfm["Segment"] = rfm["Cluster"].map(cluster_to_label)

    print("\nCluster profile (mean RFM values):")
    print(rfm.groupby("Segment")[["Recency", "Frequency", "Monetary"]].mean()
          .sort_values("Monetary", ascending=False))
    print("\nSegment sizes:")
    print(rfm["Segment"].value_counts())

    # Segment distribution chart
    fig, ax = plt.subplots(figsize=(8, 5))
    rfm["Segment"].value_counts().plot(kind="bar", ax=ax, color="#4C72B0")
    ax.set_title("Customer Segment Distribution")
    ax.set_ylabel("Number of Customers")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(EDA_DIR, "segment_distribution.png"), dpi=120)
    plt.close(fig)

    # Save artifacts
    rfm.to_csv(RFM_PATH, index=False)
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODELS_DIR, "kmeans_model.pkl"), "wb") as f:
        pickle.dump(kmeans, f)
    with open(os.path.join(MODELS_DIR, "cluster_segment_map.pkl"), "wb") as f:
        pickle.dump(cluster_to_label, f)

    print(f"\nSaved RFM table to {RFM_PATH}")
    print(f"Saved scaler, kmeans model, and cluster-segment map to {MODELS_DIR}")


if __name__ == "__main__":
    main()
