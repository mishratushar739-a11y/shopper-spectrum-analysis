"""
utils.py
--------
Shared helpers used by Home.py and every page in pages/:
    - CSS loader
    - Cached model/artifact loading
    - Recommendation + segmentation prediction logic
    - Shared constants (colors, descriptions)

Keeping this in one place means every page looks consistent and the
(possibly large) model artifacts are only loaded into memory once,
thanks to @st.cache_resource.
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
EDA_DIR = os.path.join(BASE_DIR, "outputs", "eda")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

SEGMENT_COLORS = {
    "High-Value Customers": "#2E7D32",
    "Regular Customers": "#1565C0",
    "Occasional Customers": "#F9A825",
    "At-Risk Customers": "#C62828",
}

SEGMENT_DESCRIPTIONS = {
    "High-Value Customers": "Buys often, buys recently, and spends the most. "
                             "Prioritize loyalty rewards and early access to new products.",
    "Regular Customers": "Steady, dependable buyers. Good candidates for cross-sell "
                          "and upsell campaigns.",
    "Occasional Customers": "Buys sometimes but hasn't built a strong habit yet. "
                             "Targeted promotions can increase frequency.",
    "At-Risk Customers": "Hasn't purchased in a long time. Win-back campaigns and "
                          "discounts can help re-engage them before they're lost for good.",
}


def load_css():
    """Inject the shared stylesheet into the current page."""
    css_path = os.path.join(ASSETS_DIR, "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading models...")
def load_artifacts():
    """Load every pickled artifact the app needs. Cached across pages/reruns."""
    with open(os.path.join(MODELS_DIR, "product_similarity.pkl"), "rb") as f:
        sim_df = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "product_lookup.pkl"), "rb") as f:
        product_lookup = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "description_lookup.pkl"), "rb") as f:
        description_lookup = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "kmeans_model.pkl"), "rb") as f:
        kmeans = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "cluster_segment_map.pkl"), "rb") as f:
        cluster_segment_map = pickle.load(f)
    return {
        "sim_df": sim_df,
        "product_lookup": product_lookup,
        "description_lookup": description_lookup,
        "scaler": scaler,
        "kmeans": kmeans,
        "cluster_segment_map": cluster_segment_map,
    }


def artifacts_available() -> bool:
    required = [
        "product_similarity.pkl", "product_lookup.pkl", "description_lookup.pkl",
        "scaler.pkl", "kmeans_model.pkl", "cluster_segment_map.pkl",
    ]
    return all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in required)


def show_missing_artifacts_error():
    st.error(
        "Model files were not found in the `models/` folder. Run the pipeline "
        "first, from the project root:\n\n"
        "```\npython src/data_cleaning.py\n"
        "python src/rfm_segmentation.py\n"
        "python src/recommendation.py\n```\n\n"
        "or simply:\n\n```\npython src/train_models.py\n```"
    )


def recommend_products(art: dict, product_name: str, top_n: int = 5):
    """Return up to top_n (name, score) tuples similar to product_name."""
    sim_df = art["sim_df"]
    product_lookup = art["product_lookup"]
    description_lookup = art["description_lookup"]

    key = product_name.strip().upper()

    code = description_lookup.get(key)
    if code is None:
        matches = [v for k, v in description_lookup.items() if key in k]
        code = matches[0] if matches else None

    if code is None or code not in sim_df.columns:
        return None

    scores = sim_df[code].sort_values(ascending=False)
    scores = scores.drop(labels=[code], errors="ignore").head(top_n)
    return [(product_lookup.get(c, c), float(s)) for c, s in scores.items()]


def predict_segment(art: dict, recency: float, frequency: float, monetary: float):
    scaler = art["scaler"]
    kmeans = art["kmeans"]
    cluster_segment_map = art["cluster_segment_map"]

    rfm_log = pd.DataFrame(
        [[np.log1p(recency), np.log1p(frequency), np.log1p(monetary)]],
        columns=["Recency", "Frequency", "Monetary"],
    )
    X_scaled = scaler.transform(rfm_log)
    cluster = int(kmeans.predict(X_scaled)[0])
    return cluster_segment_map.get(cluster, f"Cluster {cluster}")


def render_recommendation_list(results):
    """Render a list of (name, score) tuples as styled list items."""
    items_html = ""
    for i, (name, score) in enumerate(results, start=1):
        items_html += (
            f"<li class='rec-item'>"
            f"<span class='rec-rank'>{i}.</span>"
            f"<span class='rec-name'>{name}</span>"
            f"<span class='rec-score'>similarity {score:.2f}</span>"
            f"</li>"
        )
    st.markdown(f"<ul class='rec-list'>{items_html}</ul>", unsafe_allow_html=True)
