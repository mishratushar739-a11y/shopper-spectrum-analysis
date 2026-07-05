"""
recommendation.py
------------------
Step 4 of the Shopper Spectrum pipeline: Product Recommendation.

Builds an item-based collaborative filtering model: for every product,
we compute how similar it is to every other product based on how often
customers buy them together (cosine similarity over a customer x
product purchase-quantity matrix).

Artifacts saved to models/:
    - product_similarity.pkl   DataFrame (StockCode x StockCode) of cosine similarities
    - product_lookup.pkl       dict: StockCode -> Description (for display)
    - description_lookup.pkl   dict: normalized Description -> StockCode (for search by name)

Run directly (after data_cleaning.py):
    python src/recommendation.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
CLEAN_PATH = os.path.join(BASE_DIR, "data", "cleaned_retail.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Keep only products bought by at least this many distinct customers,
# and customers who bought at least this many distinct products, to keep
# the similarity matrix a reasonable size and remove one-off noise.
MIN_CUSTOMERS_PER_PRODUCT = 5
MIN_PRODUCTS_PER_CUSTOMER = 2


def build_purchase_matrix(df: pd.DataFrame) -> pd.DataFrame:
    product_customer_counts = df.groupby("StockCode")["CustomerID"].nunique()
    keep_products = product_customer_counts[product_customer_counts >= MIN_CUSTOMERS_PER_PRODUCT].index

    customer_product_counts = df.groupby("CustomerID")["StockCode"].nunique()
    keep_customers = customer_product_counts[customer_product_counts >= MIN_PRODUCTS_PER_CUSTOMER].index

    df_f = df[df["StockCode"].isin(keep_products) & df["CustomerID"].isin(keep_customers)]

    matrix = df_f.pivot_table(
        index="CustomerID", columns="StockCode", values="Quantity", aggfunc="sum", fill_value=0
    )
    return matrix


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = pd.read_csv(CLEAN_PATH)

    # Most common description per StockCode (descriptions can vary slightly)
    product_lookup = (
        df.groupby("StockCode")["Description"]
        .agg(lambda x: x.value_counts().idxmax())
        .to_dict()
    )

    print("Building customer-product purchase matrix...")
    matrix = build_purchase_matrix(df)
    print(f"Matrix shape: {matrix.shape[0]:,} customers x {matrix.shape[1]:,} products")

    print("Computing item-item cosine similarity...")
    item_matrix = matrix.T.values.astype(np.float32)
    sim = cosine_similarity(item_matrix)
    sim_df = pd.DataFrame(sim, index=matrix.columns, columns=matrix.columns, dtype=np.float32)

    # Build a lookup from normalized description -> stockcode, so the app
    # can accept a product NAME as input (as required by the brief).
    description_lookup = {}
    for code in sim_df.columns:
        desc = product_lookup.get(code, "")
        key = str(desc).strip().upper()
        if key and key not in description_lookup:
            description_lookup[key] = code

    with open(os.path.join(MODELS_DIR, "product_similarity.pkl"), "wb") as f:
        pickle.dump(sim_df, f)
    with open(os.path.join(MODELS_DIR, "product_lookup.pkl"), "wb") as f:
        pickle.dump(product_lookup, f)
    with open(os.path.join(MODELS_DIR, "description_lookup.pkl"), "wb") as f:
        pickle.dump(description_lookup, f)

    print(f"Saved similarity matrix ({sim_df.shape}) and lookups to {MODELS_DIR}")

    # Sanity check: show recommendations for a sample product
    sample_code = sim_df.columns[0]
    sample_name = product_lookup.get(sample_code, sample_code)
    recs = sim_df[sample_code].sort_values(ascending=False).iloc[1:6]
    print(f"\nSample: products similar to '{sample_name}' ({sample_code}):")
    for code, score in recs.items():
        print(f"  {product_lookup.get(code, code):<45} score={score:.3f}")


if __name__ == "__main__":
    main()
