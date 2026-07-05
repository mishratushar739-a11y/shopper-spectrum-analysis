"""
data_cleaning.py
-----------------
Step 1 of the Shopper Spectrum pipeline.

Loads the raw online retail transaction data and produces a clean
dataset ready for EDA, RFM segmentation, and product recommendation.

Cleaning rules applied (as required by the project brief):
    1. Remove rows with missing CustomerID (can't segment a customer we can't identify).
    2. Remove duplicate records.
    3. Remove cancelled orders (InvoiceNo starting with 'C').
    4. Remove invalid quantities (<= 0) and invalid prices (<= 0).

Run directly:
    python src/data_cleaning.py
"""

import os
import pandas as pd

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "online_retail.csv")
CLEAN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_retail.csv")


def load_raw_data(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    n_start = len(df)

    # 1. Missing values -> drop rows with no CustomerID or Description
    df = df.dropna(subset=["CustomerID", "Description"])

    # 2. Duplicates
    df = df.drop_duplicates()

    # 3. Cancelled orders (InvoiceNo starting with "C")
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

    # 4. Invalid quantities / prices
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    # Feature engineering: total price per line item
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    # Correct dtypes
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    n_end = len(df)
    print(f"Cleaned data: {n_start:,} rows -> {n_end:,} rows "
          f"({n_start - n_end:,} rows removed, {100 * (n_start - n_end) / n_start:.1f}%)")

    return df.reset_index(drop=True)


def main():
    df_raw = load_raw_data()
    df_clean = clean_data(df_raw)
    df_clean.to_csv(CLEAN_PATH, index=False)
    print(f"Saved cleaned dataset to: {CLEAN_PATH}")
    print(df_clean.describe(include="all").T)


if __name__ == "__main__":
    main()
