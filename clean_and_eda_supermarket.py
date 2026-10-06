import matplotlib
matplotlib.use("Agg")  # save charts to files, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

RAW_FILE = "supermarket_sales.csv"
CLEAN_FILE = "cleaned_supermarket_sales.csv"
CHART_DIR = "charts"

pd.set_option("display.width", 140)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
sns.set_theme(style="whitegrid")

import os
os.makedirs(CHART_DIR, exist_ok=True)


# ---------------------------------------------------------- Module 1-3: load / explore / quality
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def explore(df: pd.DataFrame) -> None:
    print("\n" + "=" * 20 + " MODULE 2: EXPLORE " + "=" * 20)
    print("Shape:", df.shape)
    df.info()
    print("\n", df.describe())


def quality_check(df: pd.DataFrame) -> None:
    print("\n" + "=" * 20 + " MODULE 3: DATA QUALITY CHECK " + "=" * 20)
    print("Missing values:\n", df.isnull().sum().to_string())
    print("Duplicate rows:", df.duplicated().sum())


# ---------------------------------------------------------- Module 4: cleaning
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Removed {before - len(df)} duplicate rows")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    bad_dates = df["Date"].isna().sum()
    if bad_dates:
        print(f"Dropping {bad_dates} rows with unreadable dates")
        df = df.dropna(subset=["Date"])

    # impute instead of dropping, so real transactions aren't lost
    missing_price = df["Unit Price"].isna().sum()
    missing_total = df["Total"].isna().sum()
    price_by_product = df.groupby("Product")["Unit Price"].transform("median")
    df["Unit Price"] = df["Unit Price"].fillna(price_by_product)
    df["Total"] = df["Total"].fillna((df["Quantity"] * df["Unit Price"]).round(2))
    print(f"Imputed {missing_price} Unit Price and {missing_total} Total values")

    df = df[(df["Quantity"] > 0) & (df["Unit Price"] > 0)].reset_index(drop=True)
    return df


# ---------------------------------------------------------- Module 5: sales statistics
def sales_statistics(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 20 + " MODULE 5: SALES STATISTICS " + "=" * 20)
    stats = {
        "Total Sales": df["Total"].sum(),
        "Average Transaction Value": df["Total"].mean(),
        "Highest Transaction": df["Total"].max(),
        "Lowest Transaction": df["Total"].min(),
        "Total Orders": len(df),
        "Total Quantity": int(df["Quantity"].sum()),
    }
    for k, v in stats.items():
        print(f"{k:<28}{v:>14,.2f}" if isinstance(v, float) else f"{k:<28}{v:>14,}")
    return stats


# ---------------------------------------------------------- Module 6-7: category sales
def category_sales(df: pd.DataFrame) -> pd.Series:
    print("\n" + "=" * 20 + " MODULE 6: SALES BY CATEGORY " + "=" * 20)
    cat = df.groupby("Category")["Total"].sum().sort_values(ascending=False)
    print(cat)
    plt.figure(figsize=(7, 4.5))
    cat.plot(kind="bar", color=sns.color_palette("Blues_r", len(cat)))
    plt.title("Sales by Category"); plt.xlabel("Category"); plt.ylabel("Total Sales ($)")
    plt.tight_layout(); plt.savefig(f"{CHART_DIR}/01_sales_by_category.png", dpi=120); plt.close()
    return cat


# ---------------------------------------------------------- Module 8: quantity by category
def quantity_by_category(df: pd.DataFrame) -> pd.Series:
    print("\n" + "=" * 20 + " MODULE 8: QUANTITY BY CATEGORY " + "=" * 20)
    qty = df.groupby("Category")["Quantity"].sum().sort_values(ascending=False)
    print(qty)
    plt.figure(figsize=(7, 4.5))
    qty.plot(kind="bar", color=sns.color_palette("Oranges_r", len(qty)))
    plt.title("Total Quantity Sold by Category"); plt.xlabel("Category"); plt.ylabel("Units Sold")
    plt.tight_layout(); plt.savefig(f"{CHART_DIR}/02_quantity_by_category.png", dpi=120); plt.close()
    return qty


# ---------------------------------------------------------- Module 9: payment methods
def payment_methods(df: pd.DataFrame) -> pd.Series:
    print("\n" + "=" * 20 + " MODULE 9: PAYMENT METHOD ANALYSIS " + "=" * 20)
    pay = df["Payment"].value_counts()
    print(pay)
    plt.figure(figsize=(6, 4.5))
    pay.plot(kind="bar", color=sns.color_palette("Greens_r", len(pay)))
    plt.title("Payment Methods"); plt.xlabel("Payment Method"); plt.ylabel("Number of Transactions")
    plt.tight_layout(); plt.savefig(f"{CHART_DIR}/03_payment_methods.png", dpi=120); plt.close()
    return pay


# ---------------------------------------------------------- Module 10: customer type
def customer_analysis(df: pd.DataFrame) -> pd.Series:
    print("\n" + "=" * 20 + " MODULE 10: CUSTOMER ANALYSIS " + "=" * 20)
    cust = df.groupby("Customer Type")["Total"].sum().sort_values(ascending=False)
    print(cust)
    plt.figure(figsize=(5, 5))
    plt.pie(cust, labels=cust.index, autopct="%1.0f%%", colors=sns.color_palette("Set2"))
    plt.title("Sales Share by Customer Type")
    plt.tight_layout(); plt.savefig(f"{CHART_DIR}/04_customer_type.png", dpi=120); plt.close()
    return cust


# ---------------------------------------------------------- Module 11-12: daily trend
def daily_sales_analysis(df: pd.DataFrame) -> pd.Series:
    print("\n" + "=" * 20 + " MODULE 11: DAILY SALES ANALYSIS " + "=" * 20)
    daily = df.groupby(df["Date"].dt.date)["Total"].sum()
    print(daily)
    plt.figure(figsize=(9, 4.5))
    daily.plot(kind="line", marker="o", color="#2a78d6")
    plt.title("Daily Sales Trend"); plt.xlabel("Date"); plt.ylabel("Sales ($)")
    plt.xticks(rotation=45); plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/05_daily_sales_trend.png", dpi=120); plt.close()

    print("\n" + "=" * 20 + " MODULE 12: HIGHEST / LOWEST SALES DAY " + "=" * 20)
    best_day, best_sales = daily.idxmax(), daily.max()
    worst_day, worst_sales = daily.idxmin(), daily.min()
    print(f"Highest Sales Day: {best_day}  -> ${best_sales:,.2f}")
    print(f"Lowest Sales Day:  {worst_day}  -> ${worst_sales:,.2f}")
    print(f"Average Daily Sales: ${daily.mean():,.2f}")
    return daily


def main() -> None:
    raw = load_data(RAW_FILE)
    explore(raw)
    quality_check(raw)
    clean = clean_data(raw.copy())
    explore(clean)

    sales_statistics(clean)
    category_sales(clean)
    quantity_by_category(clean)
    payment_methods(clean)
    customer_analysis(clean)
    daily_sales_analysis(clean)

    clean["Date"] = clean["Date"].dt.strftime("%Y-%m-%d")
    clean.to_csv(CLEAN_FILE, index=False)
    print(f"\nSaved {len(clean):,} rows to {CLEAN_FILE}")
    print(f"Charts saved to ./{CHART_DIR}/")


if __name__ == "__main__":
    main()
