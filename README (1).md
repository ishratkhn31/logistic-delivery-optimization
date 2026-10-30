# Logistics Data Analysis: Planning and Data Preparation

This repository holds the code for my data science internship tasks on logistics data analysis.

- **Week 1:** strategic plan for a last-mile delivery optimisation project
- **Week 2:** data collection, cleaning and preprocessing

The written reports for each week were submitted separately as Word documents. This repository holds the code that goes with them.

## Week 1: Strategic planning

**Scenario.** A fictional regional courier company delivers parcels from one warehouse across a city. Routes are planned by hand, which leads to late deliveries, high fuel cost and uneven vehicle loads.

**KPIs:** on-time delivery rate, cost per delivery, average distance per delivery, vehicle capacity utilisation, demand forecast error (MAPE).

**Techniques:** regression for demand forecasting, k-means clustering for delivery zones, and vehicle routing optimisation with Google OR-Tools.

**Code:** `logistics_analysis.py` shows the planned steps. By default it runs on a small synthetic dataset, so its numbers are only examples and say nothing about real logistics. It is a planning-stage script, not a finished analysis.

## Week 2: Data collection, cleaning and preprocessing

**Dataset.** DataCo Smart Supply Chain for Big Data Analysis (Kaggle). 180,519 rows and 53 columns, with orders, shipping times, customers and products.

**What `data_cleaning.py` does:**

1. Loads the file (it needs latin-1 encoding, not UTF-8)
2. Audits quality: missing values, duplicates, text problems, impossible values
3. Drops columns that are more than 50% empty and personal columns not needed for analysis
4. Strips extra spaces, converts dates, fills the few remaining missing values with the median
5. Detects outliers with the IQR rule and caps them (rows are not deleted)
6. Adds min-max and z-score scaled columns
7. Saves a cleaned file and prints a before/after summary

**Results on the real dataset:**

| Measure | Before | After |
|---|---|---|
| Rows | 180,519 | 180,519 |
| Columns | 53 | 56 |
| Missing cells | 336,209 | 0 |
| Text cells with extra spaces | 24,251 | 0 |

Outliers were found in Sales (488), Order Item Total (1,943) and Benefit per order (18,942). No duplicates or impossible values were found.

## Files

- `logistics_analysis.py`: Week 1 planning code
- `data_cleaning.py`: Week 2 cleaning pipeline
- `README.md`: this file

## How to run

```bash
pip install pandas numpy matplotlib scikit-learn ortools

# Week 1 (runs on built-in synthetic data if no file is given)
python logistics_analysis.py

# Week 2 (put the Kaggle CSV in the same folder first)
python data_cleaning.py
```

## Data

The dataset is not included in this repository because of its size. Download it from Kaggle and check its licence before use:

- DataCo Smart Supply Chain for Big Data Analysis (Constante, Silva and Pereira, 2019)
