"""
eda.py
------
Step 2 of the Shopper Spectrum pipeline: Exploratory Data Analysis.

Generates business-insight charts from the cleaned dataset and saves
them as PNG files in outputs/eda/ so they can be reviewed in VS Code
or included in a report.

Run directly (after data_cleaning.py):
    python src/eda.py
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
CLEAN_PATH = os.path.join(BASE_DIR, "data", "cleaned_retail.csv")
EDA_DIR = os.path.join(BASE_DIR, "outputs", "eda")

plt.style.use("seaborn-v0_8-whitegrid")


def load_clean_data() -> pd.DataFrame:
    df = pd.read_csv(CLEAN_PATH, parse_dates=["InvoiceDate"])
    return df


def top_selling_products(df, n=10):
    top = df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(n)
    fig, ax = plt.subplots(figsize=(9, 6))
    top.sort_values().plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_xlabel("Units Sold")
    ax.set_title(f"Top {n} Best-Selling Products")
    fig.tight_layout()
    fig.savefig(os.path.join(EDA_DIR, "top_selling_products.png"), dpi=120)
    plt.close(fig)
    return top


def country_wise_sales(df, n=10):
    sales = df.groupby("Country")["TotalPrice"].sum().sort_values(ascending=False).head(n)
    fig, ax = plt.subplots(figsize=(9, 6))
    sales.sort_values().plot(kind="barh", ax=ax, color="#55A868")
    ax.set_xlabel("Revenue")
    ax.set_title(f"Top {n} Countries by Revenue")
    fig.tight_layout()
    fig.savefig(os.path.join(EDA_DIR, "country_wise_sales.png"), dpi=120)
    plt.close(fig)
    return sales


def monthly_sales_trend(df):
    monthly = df.set_index("InvoiceDate").resample("ME")["TotalPrice"].sum()
    fig, ax = plt.subplots(figsize=(10, 5))
    monthly.plot(ax=ax, marker="o", color="#C44E52")
    ax.set_ylabel("Revenue")
    ax.set_title("Monthly Sales Trend")
    fig.tight_layout()
    fig.savefig(os.path.join(EDA_DIR, "monthly_sales_trend.png"), dpi=120)
    plt.close(fig)
    return monthly


def customer_purchase_pattern(df):
    orders_per_customer = df.groupby("CustomerID")["InvoiceNo"].nunique()
    fig, ax = plt.subplots(figsize=(9, 5))
    orders_per_customer.clip(upper=orders_per_customer.quantile(0.99)).plot(
        kind="hist", bins=40, ax=ax, color="#8172B2"
    )
    ax.set_xlabel("Number of Orders per Customer")
    ax.set_title("Customer Purchase Frequency Distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(EDA_DIR, "customer_purchase_pattern.png"), dpi=120)
    plt.close(fig)
    return orders_per_customer


def revenue_distribution(df):
    revenue_per_customer = df.groupby("CustomerID")["TotalPrice"].sum()
    fig, ax = plt.subplots(figsize=(9, 5))
    revenue_per_customer.clip(upper=revenue_per_customer.quantile(0.99)).plot(
        kind="hist", bins=40, ax=ax, color="#CCB974"
    )
    ax.set_xlabel("Total Spend per Customer")
    ax.set_title("Customer Revenue Distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(EDA_DIR, "revenue_distribution.png"), dpi=120)
    plt.close(fig)
    return revenue_per_customer


def most_active_customers(df, n=10):
    active = df.groupby("CustomerID")["InvoiceNo"].nunique().sort_values(ascending=False).head(n)
    fig, ax = plt.subplots(figsize=(9, 6))
    active.sort_values().plot(kind="barh", ax=ax, color="#64B5CD")
    ax.set_xlabel("Number of Orders")
    ax.set_title(f"Top {n} Most Active Customers")
    fig.tight_layout()
    fig.savefig(os.path.join(EDA_DIR, "most_active_customers.png"), dpi=120)
    plt.close(fig)
    return active


def main():
    os.makedirs(EDA_DIR, exist_ok=True)
    df = load_clean_data()

    print("Generating EDA charts...")
    top_selling_products(df)
    country_wise_sales(df)
    monthly_sales_trend(df)
    customer_purchase_pattern(df)
    revenue_distribution(df)
    most_active_customers(df)

    print(f"Saved 6 charts to: {EDA_DIR}")
    print("\nKey numbers:")
    print(f"  Total revenue: {df['TotalPrice'].sum():,.2f}")
    print(f"  Unique customers: {df['CustomerID'].nunique():,}")
    print(f"  Unique products: {df['StockCode'].nunique():,}")
    print(f"  Unique invoices: {df['InvoiceNo'].nunique():,}")
    print(f"  Top country by revenue: {df.groupby('Country')['TotalPrice'].sum().idxmax()}")


if __name__ == "__main__":
    main()
