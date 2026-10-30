"""
Data Collection, Cleaning and Preprocessing for Logistics Analysis (Week 2)

Reference dataset: DataCo Smart Supply Chain for Big Data Analysis (Kaggle)
File used: DataCoSupplyChainDataset.csv

What this script does:
    1. Loads the raw file and describes it
    2. Audits data quality (missing values, duplicates, data types, odd values)
    3. Cleans the data (drops, duplicates, text, dates, missing values, invalid values)
    4. Detects and treats outliers (IQR rule)
    5. Normalises numeric columns (min-max and z-score)
    6. Saves the cleaned file and prints a before/after summary

Install:  pip install pandas numpy matplotlib
Run:      python data_cleaning.py
          (or python data_cleaning.py path/to/your_file.csv)
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW_FILE = sys.argv[1] if len(sys.argv) > 1 else "DataCoSupplyChainDataset.csv"
CLEAN_FILE = "cleaned_logistics_data.csv"
MISSING_DROP_THRESHOLD = 50   # drop a column if more than this % is missing

# columns this script looks for (any that are not in your file are skipped)
DATE_COLS = ["order date (DateOrders)", "shipping date (DateOrders)"]
OUTLIER_COLS = ["Sales", "Order Item Total", "Order Item Quantity",
                "Days for shipping (real)", "Benefit per order"]
SCALE_COLS = ["Sales", "Order Item Total", "Order Item Quantity",
              "Days for shipping (real)", "Days for shipment (scheduled)"]
PRIVATE_COLS = ["Customer Email", "Customer Password", "Customer Fname",
                "Customer Lname", "Customer Street", "Product Image"]

log = {}   # keeps the numbers for the final summary


def present(cols, df):
    return [c for c in cols if c in df.columns]


# ------------------------------------------------------------ 1. collection
def load_raw(path):
    # this dataset is not UTF-8, so a different encoding is needed
    try:
        df = pd.read_csv(path, encoding="utf-8")
        enc = "utf-8"
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")
        enc = "latin-1"
    print("=" * 60)
    print("1. DATA COLLECTION")
    print("=" * 60)
    print("File:", path, "| encoding:", enc)
    print("Rows:", df.shape[0], "| Columns:", df.shape[1])
    print("\nColumn types:")
    print(df.dtypes.value_counts().to_string())
    print("\nFirst 3 rows:")
    print(df.head(3).T.head(12).to_string())
    log["raw_rows"], log["raw_cols"], log["encoding"] = df.shape[0], df.shape[1], enc
    return df


# ------------------------------------------------------------ 2. audit
def audit(df):
    print("\n" + "=" * 60)
    print("2. DATA QUALITY AUDIT (before cleaning)")
    print("=" * 60)

    # missing values
    miss = df.isna().sum()
    miss_pct = (miss / len(df) * 100).round(2)
    table = pd.DataFrame({"missing": miss, "percent": miss_pct})
    table = table[table["missing"] > 0].sort_values("percent", ascending=False)
    print("\nColumns with missing values:")
    print(table.to_string() if len(table) else "none")
    log["cols_with_missing"] = len(table)
    log["total_missing_before"] = int(miss.sum())

    if len(table):
        table["percent"].plot(kind="barh", figsize=(7, max(2, 0.4 * len(table))))
        plt.xlabel("Missing values (%)")
        plt.title("Missing values by column (before cleaning)")
        plt.tight_layout()
        plt.savefig("missing_values_before.png", dpi=120)
        plt.close()

    # duplicates
    full_dups = int(df.duplicated().sum())
    print("\nFully duplicated rows:", full_dups)
    log["duplicates"] = full_dups
    if "Order Item Id" in df.columns:
        id_dups = int(df.duplicated(subset="Order Item Id").sum())
        print("Duplicate Order Item Id values:", id_dups)
        log["id_duplicates"] = id_dups

    # text consistency
    text_cols = df.select_dtypes(include=["object", "string"]).columns
    messy = 0
    for c in text_cols:
        s = df[c].dropna().astype(str)
        messy += int((s != s.str.strip()).sum())
    print("Text cells with extra leading/trailing spaces:", messy)
    log["messy_text"] = messy

    # impossible values
    print("\nSuspicious values:")
    for c in ["Order Item Quantity", "Sales", "Order Item Total", "Days for shipping (real)"]:
        if c in df.columns:
            n = int((pd.to_numeric(df[c], errors="coerce") < 0).sum())
            print(f"  negative values in '{c}': {n}")
    if "Order Item Quantity" in df.columns:
        n = int((df["Order Item Quantity"] == 0).sum())
        print(f"  zero quantity rows: {n}")


# ------------------------------------------------------------ 3. cleaning
def clean(df):
    print("\n" + "=" * 60)
    print("3. DATA CLEANING")
    print("=" * 60)
    start = len(df)

    # a) drop columns that are mostly empty
    pct = df.isna().mean() * 100
    mostly_empty = pct[pct > MISSING_DROP_THRESHOLD].index.tolist()
    df = df.drop(columns=mostly_empty)
    print(f"a) Dropped {len(mostly_empty)} column(s) with >{MISSING_DROP_THRESHOLD}% missing:", mostly_empty)
    log["dropped_empty_cols"] = mostly_empty

    # b) drop personal columns (not useful for logistics, and a privacy risk)
    priv = present(PRIVATE_COLS, df)
    df = df.drop(columns=priv)
    print("b) Dropped personal / non-analytical columns:", priv)
    log["dropped_private_cols"] = priv

    # c) remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    if "Order Item Id" in df.columns:
        df = df.drop_duplicates(subset="Order Item Id")
    print(f"c) Removed {before - len(df)} duplicate row(s)")
    log["dups_removed"] = before - len(df)

    # d) tidy text
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].astype("string").str.strip()
    print("d) Stripped extra spaces from text columns")

    # e) dates
    for c in present(DATE_COLS, df):
        df[c] = pd.to_datetime(df[c], errors="coerce")
        print(f"e) Converted '{c}' to datetime ({int(df[c].isna().sum())} unparseable)")
    d1, d2 = [c for c in DATE_COLS if c in df.columns][:2] if len(present(DATE_COLS, df)) == 2 else (None, None)
    if d1 and d2:
        bad = int((df[d2] < df[d1]).sum())
        print(f"   rows where shipping date is before order date: {bad}")
        df = df[~(df[d2] < df[d1])]
        log["bad_dates_removed"] = bad

    # f) invalid values
    inv = 0
    if "Order Item Quantity" in df.columns:
        mask = df["Order Item Quantity"] <= 0
        inv += int(mask.sum())
        df = df[~mask]
    if "Days for shipping (real)" in df.columns:
        mask = df["Days for shipping (real)"] < 0
        inv += int(mask.sum())
        df = df[~mask]
    print(f"f) Removed {inv} row(s) with impossible values (zero/negative quantity or days)")
    log["invalid_removed"] = inv

    # g) remaining missing values
    filled_num, filled_cat = 0, 0
    for c in df.columns:
        n = int(df[c].isna().sum())
        if n == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(df[c].median())      # median: not pulled by outliers
            filled_num += n
        elif pd.api.types.is_datetime64_any_dtype(df[c]):
            continue
        else:
            df[c] = df[c].fillna("Unknown")
            filled_cat += n
    print(f"g) Filled {filled_num} missing numeric cell(s) with the median "
          f"and {filled_cat} missing text cell(s) with 'Unknown'")
    log["filled_num"], log["filled_cat"] = filled_num, filled_cat

    # h) a useful derived column
    if {"Days for shipping (real)", "Days for shipment (scheduled)"} <= set(df.columns):
        df["shipping_delay_days"] = df["Days for shipping (real)"] - df["Days for shipment (scheduled)"]
        print("h) Added 'shipping_delay_days' = real days - scheduled days")

    print(f"\nRows: {start} -> {len(df)}  (removed {start - len(df)})")
    log["rows_after_cleaning"] = len(df)
    return df.reset_index(drop=True)


# ------------------------------------------------------------ 4. outliers
def treat_outliers(df):
    print("\n" + "=" * 60)
    print("4. OUTLIER DETECTION (IQR rule)")
    print("=" * 60)
    cols = present(OUTLIER_COLS, df)
    if cols:
        df[cols].boxplot(figsize=(9, 4), rot=20)
        plt.title("Boxplots before outlier treatment")
        plt.tight_layout()
        plt.savefig("outliers_before.png", dpi=120)
        plt.close()

    log["outliers"] = {}
    for c in cols:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((df[c] < low) | (df[c] > high)).sum())
        print(f"{c}: {n} outlier(s)  (valid range {low:.2f} to {high:.2f})")
        # capping instead of deleting, so no rows are lost
        df[c] = df[c].clip(lower=low, upper=high)
        log["outliers"][c] = n
    print("\nOutliers were capped at the IQR limits (not deleted), so row count is unchanged.")
    return df


# ------------------------------------------------------------ 5. normalisation
def normalise(df):
    print("\n" + "=" * 60)
    print("5. NORMALISATION")
    print("=" * 60)
    cols = present(SCALE_COLS, df)
    for c in cols:
        rng = df[c].max() - df[c].min()
        df[c + "_minmax"] = (df[c] - df[c].min()) / rng if rng else 0.0
        std = df[c].std()
        df[c + "_zscore"] = (df[c] - df[c].mean()) / std if std else 0.0
    print("Added min-max (0 to 1) and z-score columns for:", cols)
    if cols:
        print("\nExample (first 5 rows, min-max):")
        print(df[[c + "_minmax" for c in cols]].head().round(3).to_string())
    return df


# ------------------------------------------------------------ 6. summary
def summary(df_clean):
    print("\n" + "=" * 60)
    print("6. FINAL SUMMARY  (copy everything from here into the chat)")
    print("=" * 60)
    print("Encoding used:", log.get("encoding"))
    print("Raw rows x columns:", log.get("raw_rows"), "x", log.get("raw_cols"))
    print("Columns with missing values:", log.get("cols_with_missing"))
    print("Total missing cells before:", log.get("total_missing_before"))
    print("Fully duplicated rows:", log.get("duplicates"))
    print("Duplicate Order Item Id:", log.get("id_duplicates"))
    print("Messy text cells:", log.get("messy_text"))
    print("Columns dropped (mostly empty):", log.get("dropped_empty_cols"))
    print("Columns dropped (personal):", log.get("dropped_private_cols"))
    print("Duplicates removed:", log.get("dups_removed"))
    print("Bad date rows removed:", log.get("bad_dates_removed"))
    print("Invalid-value rows removed:", log.get("invalid_removed"))
    print("Numeric cells filled (median):", log.get("filled_num"))
    print("Text cells filled ('Unknown'):", log.get("filled_cat"))
    print("Outliers capped:", log.get("outliers"))
    print("Rows after cleaning:", log.get("rows_after_cleaning"))
    print("Columns after cleaning:", df_clean.shape[1])
    print("Missing cells after cleaning:", int(df_clean.isna().sum().sum()))


if __name__ == "__main__":
    try:
        raw = load_raw(RAW_FILE)
    except FileNotFoundError:
        sys.exit(f"Could not find '{RAW_FILE}'. Put the CSV in the same folder "
                 "or pass its path: python data_cleaning.py path/to/file.csv")
    audit(raw)
    cleaned = clean(raw.copy())
    cleaned = treat_outliers(cleaned)
    cleaned = normalise(cleaned)
    cleaned.to_csv(CLEAN_FILE, index=False)
    print(f"\nSaved cleaned data to {CLEAN_FILE}")
    summary(cleaned)
