"""
Exploratory Data Analysis and Visualization for Logistics Data (Week 3)

Input : cleaned_logistics_data.csv  (created by data_cleaning.py in Week 2)
Output: summary statistics printed to the screen and 8 chart images (PNG)

Install:  pip install pandas numpy matplotlib seaborn
Run:      python eda_visualization.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

CLEAN_FILE = sys.argv[1] if len(sys.argv) > 1 else "cleaned_logistics_data.csv"
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

LATE = "Late_delivery_risk"
REAL = "Days for shipping (real)"
SCHED = "Days for shipment (scheduled)"
DELAY = "shipping_delay_days"
MODE = "Shipping Mode"
MARKET = "Market"
REGION = "Order Region"
CATEGORY = "Category Name"
DATE = "order date (DateOrders)"


def has(df, *cols):
    return all(c in df.columns for c in cols)


def save(name):
    plt.tight_layout()
    plt.savefig(name, dpi=150)
    plt.close()
    print("   saved", name)


def title(text):
    print("\n" + "=" * 62)
    print(text)
    print("=" * 62)


# ---------------------------------------------------------------- load
if not os.path.exists(CLEAN_FILE):
    sys.exit(f"Could not find '{CLEAN_FILE}'. Run data_cleaning.py first "
             "(it creates this file) or pass the file path.")

df = pd.read_csv(CLEAN_FILE, low_memory=False)
if DATE in df.columns:
    df[DATE] = pd.to_datetime(df[DATE], errors="coerce")

title("1. DATASET OVERVIEW")
print("Rows:", len(df), "| Columns:", df.shape[1])
if DATE in df.columns:
    print("Order dates from", df[DATE].min().date(), "to", df[DATE].max().date())

# ---------------------------------------------------------------- statistics
title("2. CENTRAL TENDENCY AND SPREAD")
key_num = [c for c in ["Sales", "Order Item Total", "Order Item Quantity",
                       REAL, SCHED, DELAY, "Benefit per order"] if c in df.columns]
stats = df[key_num].agg(["mean", "median", "std", "min", "max", "skew"]).T.round(2)
print(stats.to_string())

title("3. DELIVERY PERFORMANCE")
if LATE in df.columns:
    late_rate = df[LATE].mean() * 100
    print(f"Overall late delivery rate: {late_rate:.1f}%  (on time: {100 - late_rate:.1f}%)")
if "Delivery Status" in df.columns:
    print("\nDelivery status share (%):")
    print((df["Delivery Status"].value_counts(normalize=True) * 100).round(1).to_string())
if has(df, MODE, LATE):
    print("\nLate delivery rate by shipping mode (%):")
    print((df.groupby(MODE)[LATE].mean() * 100).round(1).sort_values(ascending=False).to_string())
if has(df, MODE, REAL, SCHED):
    print("\nAverage real vs scheduled shipping days by mode:")
    print(df.groupby(MODE)[[REAL, SCHED]].mean().round(2).to_string())
if has(df, MARKET, LATE):
    print("\nLate delivery rate by market (%):")
    print((df.groupby(MARKET)[LATE].mean() * 100).round(1).sort_values(ascending=False).to_string())
if has(df, REGION, LATE):
    r = (df.groupby(REGION)[LATE].mean() * 100).round(1).sort_values(ascending=False)
    print("\nHighest and lowest late rate by order region (%):")
    print("  highest:", r.head(3).to_dict())
    print("  lowest :", r.tail(3).to_dict())

title("4. SALES AND PRODUCTS")
if has(df, MARKET, "Sales"):
    print("Total sales by market:")
    print(df.groupby(MARKET)["Sales"].sum().round(0).sort_values(ascending=False).to_string())
if has(df, CATEGORY, "Sales"):
    print("\nTop 5 categories by sales:")
    print(df.groupby(CATEGORY)["Sales"].sum().round(0).sort_values(ascending=False).head(5).to_string())
if DATE in df.columns:
    monthly = df.groupby(df[DATE].dt.to_period("M")).size()
    print("\nOrders per month: average", round(monthly.mean()), "| busiest", monthly.idxmax(),
          "(", monthly.max(), ") | quietest", monthly.idxmin(), "(", monthly.min(), ")")

title("5. CORRELATIONS")
corr_cols = [c for c in ["Sales", "Order Item Total", "Order Item Quantity", "Benefit per order",
                         "Order Item Discount", "Order Item Discount Rate",
                         REAL, SCHED, DELAY, LATE] if c in df.columns]
corr = df[corr_cols].corr().round(2)
if LATE in corr.columns:
    print("Correlation with late delivery risk:")
    print(corr[LATE].drop(LATE).sort_values(ascending=False).to_string())
pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().sort_values(key=abs, ascending=False)
print("\nStrongest pairs:")
print(pairs.head(5).to_string())

# ---------------------------------------------------------------- charts
title("6. CREATING CHARTS")

# 1 distribution of real shipping days
if REAL in df.columns:
    plt.figure(figsize=(7, 4.5))
    sns.histplot(df[REAL], bins=range(0, int(df[REAL].max()) + 2), color="#2a6f97")
    plt.axvline(df[REAL].mean(), color="red", linestyle="--", label=f"Mean = {df[REAL].mean():.2f}")
    plt.xlabel("Actual shipping days"); plt.ylabel("Number of order items")
    plt.title("Distribution of actual shipping days"); plt.legend()
    save("chart1_shipping_days_distribution.png")

# 2 late rate by shipping mode
if has(df, MODE, LATE):
    plt.figure(figsize=(7, 4.5))
    s = (df.groupby(MODE)[LATE].mean() * 100).sort_values(ascending=False)
    ax = sns.barplot(x=s.index, y=s.values, color="#c0392b")
    for i, v in enumerate(s.values):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center")
    plt.ylabel("Late deliveries (%)"); plt.xlabel("")
    plt.title("Late delivery rate by shipping mode"); plt.ylim(0, s.max() + 12)
    save("chart2_late_rate_by_shipping_mode.png")

# 3 delivery status
if "Delivery Status" in df.columns:
    plt.figure(figsize=(7, 4.5))
    s = df["Delivery Status"].value_counts()
    ax = sns.barplot(x=s.index, y=s.values, color="#52796f")
    for i, v in enumerate(s.values):
        ax.text(i, v + s.max() * 0.01, f"{v:,}", ha="center")
    plt.ylabel("Number of order items"); plt.xlabel("")
    plt.title("Delivery status of all orders")
    plt.xticks(rotation=10)
    save("chart3_delivery_status.png")

# 4 sales by market
if has(df, MARKET, "Sales"):
    plt.figure(figsize=(7, 4.5))
    s = df.groupby(MARKET)["Sales"].sum().sort_values(ascending=False) / 1e6
    sns.barplot(x=s.index, y=s.values, color="#6a4c93")
    plt.ylabel("Total sales (millions)"); plt.xlabel("")
    plt.title("Total sales by market")
    save("chart4_sales_by_market.png")

# 5 orders over time
if DATE in df.columns:
    plt.figure(figsize=(9, 4.5))
    m = df.groupby(df[DATE].dt.to_period("M")).size()
    m.index = m.index.to_timestamp()
    plt.plot(m.index, m.values, marker="o", markersize=3, color="#1d3557")
    plt.ylabel("Order items per month"); plt.xlabel("Month")
    plt.title("Monthly order volume over time")
    save("chart5_orders_over_time.png")

# 6 correlation heatmap
if len(corr_cols) > 2:
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, annot_kws={"size": 8})
    plt.title("Correlation between key variables")
    save("chart6_correlation_heatmap.png")

# 7 boxplot shipping days by mode
if has(df, MODE, REAL):
    plt.figure(figsize=(7, 4.5))
    sns.boxplot(data=df, x=MODE, y=REAL, color="#8ecae6")
    plt.xlabel(""); plt.ylabel("Actual shipping days")
    plt.title("Actual shipping days by shipping mode")
    save("chart7_shipping_days_by_mode.png")

# 8 top categories
if has(df, CATEGORY, "Sales"):
    plt.figure(figsize=(8, 5))
    s = (df.groupby(CATEGORY)["Sales"].sum().sort_values(ascending=False).head(10) / 1e6)[::-1]
    plt.barh(s.index, s.values, color="#e07a5f")
    plt.xlabel("Total sales (millions)")
    plt.title("Top 10 product categories by sales")
    save("chart8_top_categories.png")

print("\nDone. Copy the printed output above and download the chart1 to chart8 PNG files.")
