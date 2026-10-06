import numpy as np
import pandas as pd

SEED = 21
N_BASE = 1000
YEAR, MONTH = 2023, 1
OUTPUT_FILE = "supermarket_sales.csv"

rng = np.random.default_rng(SEED)

# product: (category, min price, max price)
PRODUCTS = {
    "Bread":        ("Food", 1.5, 3.0),
    "Rice":         ("Food", 3.0, 8.0),
    "Eggs":         ("Food", 2.0, 4.5),
    "Pasta":        ("Food", 1.5, 4.0),
    "Milk":         ("Beverages", 1.0, 2.5),
    "Juice":        ("Beverages", 2.0, 5.0),
    "Soda":         ("Beverages", 1.0, 3.0),
    "Coffee":       ("Beverages", 4.0, 9.0),
    "Earphones":    ("Electronics", 8.0, 25.0),
    "Batteries":    ("Electronics", 3.0, 10.0),
    "Extension Cord": ("Electronics", 6.0, 15.0),
    "T-Shirt":      ("Clothing", 6.0, 15.0),
    "Socks":        ("Clothing", 2.0, 6.0),
    "Detergent":    ("Household", 4.0, 9.0),
    "Tissue Paper": ("Household", 1.5, 4.0),
    "Dish Soap":    ("Household", 2.0, 5.0),
}
CUSTOMER_TYPES = ["Member", "Normal"]
PAYMENTS = ["Cash", "Credit Card", "Debit Card", "Digital Payment"]
PAYMENT_WEIGHTS = [0.40, 0.30, 0.20, 0.10]


def build_clean_orders() -> pd.DataFrame:
    names = list(PRODUCTS)
    days_in_month = pd.Period(f"{YEAR}-{MONTH:02d}").days_in_month
    dates = pd.date_range(f"{YEAR}-{MONTH:02d}-01", periods=days_in_month)

    # weekend demand boost (Sat/Sun)
    day_weight = np.array([1.3 if d.weekday() >= 5 else 1.0 for d in dates])
    day_probs = day_weight / day_weight.sum()
    day_choice = rng.choice(dates, size=N_BASE, p=day_probs)

    product = rng.choice(names, size=N_BASE)
    category = [PRODUCTS[p][0] for p in product]
    cust_type = rng.choice(CUSTOMER_TYPES, size=N_BASE, p=[0.62, 0.38])
    qty_boost = np.where(cust_type == "Member", 1, 0)
    quantity = rng.integers(1, 6, size=N_BASE) + qty_boost
    quantity = np.clip(quantity, 1, 10)

    unit_price = np.array([round(rng.uniform(*PRODUCTS[p][1:]), 2) for p in product])
    payment = rng.choice(PAYMENTS, size=N_BASE, p=PAYMENT_WEIGHTS)

    df = pd.DataFrame({
        "Date": day_choice,
        "Product": product,
        "Category": category,
        "Quantity": quantity,
        "Unit Price": unit_price,
        "Customer Type": cust_type,
        "Payment": payment,
    })
    df["Total"] = (df["Quantity"] * df["Unit Price"]).round(2)
    df = df.sort_values("Date").reset_index(drop=True)
    df.insert(0, "Invoice ID", [f"INV-{1001 + i}" for i in range(len(df))])
    return df


def add_dirt(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")  # store as text, like a real export

    n_missing = int(0.015 * len(df))
    df.loc[rng.choice(df.index, n_missing, replace=False), "Unit Price"] = np.nan
    df.loc[rng.choice(df.index, n_missing, replace=False), "Total"] = np.nan

    dups = df.sample(int(0.01 * len(df)), random_state=SEED)
    df = pd.concat([df, dups]).sort_values("Invoice ID", kind="stable").reset_index(drop=True)
    return df


if __name__ == "__main__":
    raw = add_dirt(build_clean_orders())
    raw.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(raw):,} rows to {OUTPUT_FILE}")
    print(raw.head(8).to_string(index=False))
    print("\nMissing values:\n", raw.isna().sum().to_string())
    print("\nDuplicate rows:", raw.duplicated().sum())
